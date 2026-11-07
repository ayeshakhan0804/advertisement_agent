import time
import streamlit as st

from agent import run
from image_gen import generate_image

st.set_page_config(page_title="Multi-Agent Advertising AI System", page_icon="📣", layout="wide")
st.title("📣 Autonomous Multi-Agent Advertising System")
st.caption("Advanced Multi-Agent System: Planner ➔ Researcher ➔ Tools ➔ Copywriter ➔ Policy Validator ➔ Critic Reflection")

if "brief" not in st.session_state:
    st.session_state.brief = None
    st.session_state.final = None
    st.session_state.trace = []

with st.sidebar:
    st.header("Campaign Brief")
    product = st.text_input("Product / Service", placeholder="e.g., EcoBottle, SaaS CRM")
    audience = st.text_input("Target Audience", placeholder="e.g., fitness enthusiasts, startups")
    platform = st.selectbox("Ad Platform", ["instagram", "google_ads", "facebook", "twitter", "linkedin"])
    tone = st.selectbox("Tone", ["friendly", "professional", "funny", "luxury", "urgent"])
    budget = st.number_input("Budget in $ (optional)", min_value=0.0, value=500.0, step=50.0)
    price = st.number_input("Product Price in $", min_value=0.0, value=25.0, step=5.0)
    go = st.button("🚀 Run Agentic Workflow", type="primary", use_container_width=True)

    st.divider()
    with st.expander("📋 M.Tech Rubric & Architecture", expanded=False):
        st.markdown("""
        **1. Agent Workflow & Planning (5 Marks)**
        - LangGraph cyclical state graph.
        - Decomposes brief into 4-5 numbered execution steps.
        
        **2. Tool Calling & Execution (5 Marks)**
        - External Web Search API (`ddgs`)
        - Platform Policy & Spec Checker
        - SQLite Database Connector (`campaigns.db`)
        - Financial / ROAS Calculator Engine
        - Hashtag Extraction Tool
        - Pollinations AI Visual Generator
        
        **3. State Management & Self-Correction (4 Marks)**
        - Tool error fallback & graceful retry
        - Platform character limit self-correction loop
        - Strict policy check (banned claims / emojis)
        - LLM Critic reflection & iterative scoring
        
        **4. Live Working Demo (6 Marks)**
        - Full live trace inspection
        - Real-time parameter adaptation & follow-up chat
        """)


def execute(brief):
    log_box = st.status("Agent executing multi-step workflow...", expanded=True)
    final = None
    accumulated_trace = []
    try:
        for node, upd in run(brief):
            for line in upd.get("trace", []):
                if node == "tools" or line.startswith("TOOL CALL") or line.startswith("TOOL RESULT"):
                    continue
                accumulated_trace.append((node, line))
                log_box.markdown(f"**`[{node.upper()}]`**")
                log_box.code(line, language=None)
            final = upd.get("final", final)
    except Exception as e:
        log_box.update(label="Agent execution failed", state="error", expanded=True)
        st.error(f"{type(e).__name__}: {e}")
        return None, accumulated_trace
    
    if final:
        log_box.update(label="Agent execution succeeded! (Inspect full trace above)", state="complete", expanded=False)
    else:
        log_box.update(label="Agent halted: brief requirements incomplete", state="error", expanded=True)
    return final, accumulated_trace


def show_image_fallback(prompt):
    cache = st.session_state.setdefault("images", {})
    if prompt in cache:
        st.image(cache[prompt], caption="Regenerated Ad Creative", use_container_width=True)
        return
    if prompt and st.button("🔄 Regenerate Alternative Creative", key="img_btn"):
        try:
            with st.spinner("Generating image..."):
                cache[prompt] = generate_image(prompt, st.session_state.get("last_img", 0.0))
            st.session_state.last_img = time.time()
            st.rerun()
        except Exception as e:
            st.warning(str(e))


def show(final):
    ad = final["ad"]
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(ad.get("headline", "Generated Ad"))
        st.write(ad.get("body", ""))
        st.markdown(f"**CTA:** `{ad.get('cta', 'Learn More')}`")
        if ad.get("alt_headline"):
            st.caption(f"🅰️/🅱️ **A/B Test Headline:** {ad['alt_headline']}")
        if ad.get("hashtags"):
            st.write(" ".join(ad.get("hashtags", [])))
        
        # Display Generated Visual Creative
        if ad.get("image_url"):
            st.image(ad["image_url"], caption="🎨 AI-Generated Ad Creative (Pollinations FLUX/SDXL)", use_container_width=True)
        elif ad.get("image_prompt"):
            st.info(f"🎨 Image prompt: {ad.get('image_prompt', '')}")
            show_image_fallback(ad.get("image_prompt", ""))
    
    with c2:
        st.metric("Draft Iterations Used", final.get("drafts_used", 1))
        issues = final.get("remaining_issues", [])
        st.metric("Policy Validation", "Passed ✅" if not issues else "Issues Left ⚠️")
        if final.get("critic_score") is not None:
            st.metric("Critic Quality Score", f"{final['critic_score']}/10")
        if final.get("forecast"):
            st.markdown("### 📊 Financial & ROAS Forecast")
            st.json(final["forecast"])


# Handle Main Trigger
if go:
    brief = {
        "product": product,
        "audience": audience,
        "platform": platform,
        "tone": tone,
        "budget": budget if budget > 0 else None,
        "price": price
    }
    st.session_state.brief = brief
    final, trace = execute(brief)
    st.session_state.final = final
    st.session_state.trace = trace

elif st.session_state.get("final"):
    # Show existing execution if available
    pass

# Display Result & Conversation History
if st.session_state.get("final"):
    show(st.session_state["final"])
    
    # Follow-up interactive refinement chat
    st.divider()
    st.markdown("### 💬 Interactive Refinement (Context Retention)")
    follow = st.chat_input("Refine this campaign (e.g., 'Make it punchier', 'Focus on 20% discount', 'Emphasize durability')")
    if follow:
        refined_brief = {
            **st.session_state.brief,
            "instruction": follow,
            "previous_ad": st.session_state.final["ad"]
        }
        st.session_state.brief = refined_brief
        final, trace = execute(refined_brief)
        st.session_state.final = final
        st.session_state.trace = trace
        st.rerun()

elif go and not st.session_state.get("final"):
    st.warning("⚠️ No campaign could be generated. Please ensure Product and Audience are provided.")
