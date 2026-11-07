"""Advertisement Agent - LangGraph state graph with tool calling, validation and self-correction.

Flow:  planner -> researcher <-> tools -> writer -> validator
       validator --(rule issues)--> writer | validator --(ok)--> critic --(score<7)--> writer
       critic --(score>=7)--> finalizer -> END   (max 3 drafts)
"""
import json
import operator
import os
import re
import sys
from typing import Annotated, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import tools as T

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_TOOL_ROUNDS = 6
MAX_DRAFTS = 3
TOOLS_BY_NAME = {t.name: t for t in T.RESEARCH_TOOLS}


class AdState(TypedDict, total=False):
    brief: dict
    plan: str
    messages: Annotated[list, add_messages]
    tool_rounds: int
    draft: dict
    issues: list
    drafts_made: int
    critic_score: int
    final: dict
    error: str
    trace: Annotated[list, operator.add]


# ---------- helpers ----------
def llm(temperature=0.7):
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY is missing. Put it in a .env file next to agent.py.")
    # fail fast instead of retrying silently, so the UI can show the real error
    return ChatGoogleGenerativeAI(model=MODEL, temperature=temperature, max_retries=3, timeout=60)


def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in c)
    return c or ""


def parse_json(s: str) -> dict:
    s = re.sub(r"```(?:json)?", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(s[start:end + 1])


def brief_text(b: dict) -> str:
    lines = [f"Product: {b.get('product')}", f"Target audience: {b.get('audience')}",
             f"Platform: {b.get('platform')}", f"Tone: {b.get('tone', 'friendly')}"]
    if b.get("budget"):
        lines.append(f"Budget: {b['budget']}")
    if b.get("instruction"):
        lines.append(f"FOLLOW-UP REQUEST from user: {b['instruction']}")
        lines.append(f"Previous ad to improve: {json.dumps(b.get('previous_ad', {}))}")
    return "\n".join(lines)


# ---------- nodes ----------
def planner(state: AdState):
    b = state["brief"]
    if not str(b.get("product", "")).strip() or not str(b.get("audience", "")).strip():
        msg = "Brief incomplete: please give at least the product and the target audience."
        return {"error": msg, "trace": [f"STOP: {msg}"]}
    tool_list = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in T.RESEARCH_TOOLS)
    prompt = ("You are the planner of an advertising agent. Break this brief into 4-5 numbered steps, "
              "ONE short line each (max 15 words). Every research step must name exactly one of the agent's "
              "REAL tools below; the last steps are 'write ad copy', 'validate against platform rules' "
              "and 'critic review'. Do NOT mention any other software.\n\nREAL TOOLS:\n" + tool_list +
              "\n\n" + brief_text(b))
    plan = text_of(llm(0.3).invoke(prompt))
    system = SystemMessage(content=(
        "You are the research stage of an advertising agent. Use tools to gather what you need: "
        "get_platform_specs (ALWAYS), web_search (trends/competitors), get_past_campaigns, "
        "generate_hashtags, and estimate_campaign_budget if a budget is given. "
        "If a tool returns ERROR, read the message and retry with corrected arguments or a different tool. "
        "When you have enough, reply with a short research summary and NO tool calls."))
    human = HumanMessage(content=f"BRIEF:\n{brief_text(b)}\n\nPLAN:\n{plan}")
    return {"plan": plan, "messages": [system, human], "tool_rounds": 0, "drafts_made": 0,
            "issues": [], "trace": [f"PLAN:\n{plan}"]}


def researcher(state: AdState):
    resp = llm(0.2).bind_tools(T.RESEARCH_TOOLS).invoke(state["messages"])
    if resp.tool_calls:
        trace = []
    else:
        trace = [f"RESEARCH SUMMARY: {text_of(resp)[:600]}"]
    return {"messages": [resp], "trace": trace}


def tool_node(state: AdState):
    last, out = state["messages"][-1], []
    for tc in last.tool_calls:
        try:
            fn = TOOLS_BY_NAME.get(tc["name"])
            if fn is None:
                raise ValueError(f"Unknown tool '{tc['name']}'. Available: {', '.join(TOOLS_BY_NAME)}")
            result = fn.invoke(tc["args"])
        except Exception as e:  # self-correction: hand the error back to the LLM instead of crashing
            result = (f"ERROR: {type(e).__name__}: {e}. Retry with corrected arguments, "
                      f"or continue without this tool and say so.")
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": out, "tool_rounds": state["tool_rounds"] + 1, "trace": []}


def route_research(state: AdState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state["tool_rounds"] < MAX_TOOL_ROUNDS:
        return "tools"
    return "writer"


def writer(state: AdState):
    b, n = state["brief"], state["drafts_made"] + 1
    research = "\n".join(f"[{m.name}] {m.content}" for m in state["messages"] if isinstance(m, ToolMessage))
    feedback = ""
    if state.get("issues"):
        feedback = (f"\n\nYour previous draft FAILED validation:\n- " + "\n- ".join(state["issues"]) +
                    f"\nPrevious draft: {json.dumps(state['draft'])}\nFix EVERY issue and keep within the limits.")
    prompt = (f"You are an expert ad copywriter. Write ONE ad.\n\n{brief_text(b)}\n\n"
              f"RESEARCH:\n{research or 'none'}{feedback}\n\n"
              "Respond with ONLY a JSON object with keys: headline, body, cta, hashtags (list of strings), "
              "alt_headline (an A/B-test alternative headline within the limit), "
              "image_prompt (a vivid, detailed visual description for generating an ad creative that matches the product, script, and tone, commercial photography style, no text in image). Respect the platform's character limits.")
    draft, err = None, ""
    for _ in range(2):  # self-correction for malformed JSON
        raw = text_of(llm(0.8).invoke(prompt + err))
        try:
            draft = parse_json(raw)
            break
        except Exception as e:
            err = f"\n\nYour last reply was not valid JSON ({e}). Return ONLY the JSON object."
    if draft is None:
        draft = {"headline": "", "body": "", "cta": "", "hashtags": [], "image_prompt": ""}
    for k in ("headline", "body", "cta"):
        draft[k] = str(draft.get(k, ""))
    return {"draft": draft, "drafts_made": n,
            "trace": [f"DRAFT #{n}: {draft['headline']} | {draft['body']} | CTA: {draft['cta']}"]}


def validator(state: AdState):
    d, b = state["draft"], state["brief"]
    res = json.loads(T.check_ad_copy(b.get("platform", ""), d["headline"], d["body"], d["cta"]))
    status = "PASSED" if res["ok"] else f"FAILED -> {res['issues']}"
    return {"issues": res["issues"], "trace": [f"VALIDATION (draft {state['drafts_made']}): {status}"]}


def route_validation(state: AdState):
    if state["issues"]:
        return "writer" if state["drafts_made"] < MAX_DRAFTS else "finalizer"
    return "critic"


def critic(state: AdState):
    """LLM reflection: scores the draft against the brief and sends it back if it is weak."""
    d, b = state["draft"], state["brief"]
    prompt = (f"You are a strict advertising critic.\n{brief_text(b)}\n\nAD:\n{json.dumps(d)}\n\n"
              "Score 1-10 for: hook strength, fit with audience and tone, clarity of benefit, CTA. "
              'Reply ONLY with JSON: {"score": <int>, "feedback": "<one concrete fix>"}')
    try:
        r = parse_json(text_of(llm(0.2).invoke(prompt)))
        score, feedback = int(r.get("score", 10)), str(r.get("feedback", ""))
    except Exception:  # fallback: if the critic fails, accept the draft instead of crashing
        return {"issues": [], "critic_score": None, "trace": ["CRITIC: unavailable, accepting draft (fallback)"]}
    low = score < 7 and state["drafts_made"] < MAX_DRAFTS
    return {"issues": [f"Critic feedback (score {score}/10): {feedback}"] if low else [],
            "critic_score": score,
            "trace": [f"CRITIC: {score}/10 - {feedback}" + (" -> sending back for revision" if low else " -> accepted")]}


def route_critic(state: AdState):
    return "writer" if state["issues"] else "finalizer"


def finalizer(state: AdState):
    b, ad = state["brief"], state["draft"]
    forecast = None
    if b.get("budget"):
        try:
            clean_b = float(re.sub(r"[^\d.]", "", str(b["budget"])))
            forecast = json.loads(T.estimate_campaign_budget(
                clean_b, float(b.get("cpc", 0.5)),
                float(b.get("conv_rate", 2)), float(b.get("price", 50))))
        except Exception as e:
            forecast = {"error": str(e)}

    img_trace = []
    if ad.get("image_prompt"):
        try:
            img_url = T.generate_ad_image(ad["image_prompt"], b.get("platform", "instagram"))
            ad["image_url"] = img_url
            img_trace.append(f"IMAGE GENERATED: {img_url}")
        except Exception as e:
            ad["image_url"] = None
            img_trace.append(f"IMAGE GENERATION WARNING: {e}")

    T.save_campaign(b["product"], b["platform"], ad)
    final = {"ad": ad, "forecast": forecast, "remaining_issues": state["issues"],
             "drafts_used": state["drafts_made"],
             "critic_score": state.get("critic_score")}
    return {"final": final, "trace": ["SAVED campaign to database."] + img_trace + ["DONE."]}


# ---------- graph ----------
def build_graph():
    g = StateGraph(AdState)
    for name, fn in [("planner", planner), ("researcher", researcher), ("tools", tool_node),
                     ("writer", writer), ("validator", validator), ("critic", critic),
                     ("finalizer", finalizer)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", lambda s: "end" if s.get("error") else "researcher",
                            {"end": END, "researcher": "researcher"})
    g.add_conditional_edges("researcher", route_research, {"tools": "tools", "writer": "writer"})
    g.add_edge("tools", "researcher")
    g.add_edge("writer", "validator")
    g.add_conditional_edges("validator", route_validation,
                            {"writer": "writer", "critic": "critic", "finalizer": "finalizer"})
    g.add_conditional_edges("critic", route_critic, {"writer": "writer", "finalizer": "finalizer"})
    g.add_edge("finalizer", END)
    return g.compile()


def run(brief: dict):
    """Yield (node_name, update_dict) as the graph executes."""
    for update in build_graph().stream({"brief": brief}, stream_mode="updates"):
        for node, upd in update.items():
            yield node, upd


if __name__ == "__main__":
    brief = {"product": input("Product: "), "audience": input("Audience: "),
             "platform": input("Platform (google_ads/facebook/instagram/twitter/linkedin): "),
             "tone": input("Tone [friendly]: ") or "friendly", "budget": input("Budget (optional): ") or None}
    for node, upd in run(brief):
        for line in upd.get("trace", []):
            print(f"\n[{node}] {line}")
        if "final" in upd:
            print("\n=== FINAL ===\n", json.dumps(upd["final"], indent=2))