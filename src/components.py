import streamlit as st
from .constants import POLICY_QA

def header(title, goal):
    st.title(title)
    st.markdown(f"> **Mục tiêu kinh tế:** {goal}")
    st.markdown("---")

def policy_box(bai: int):
    """Hiển thị mục 'Câu hỏi thảo luận chính sách' cho từng bài (mở rộng được)."""
    qa = POLICY_QA.get(bai)
    if not qa:
        return
    st.markdown("---")
    with st.expander(f"💬 Câu hỏi thảo luận chính sách — Bài {bai}", expanded=False):
        st.caption(
            "Phản biện chính sách gắn kết quả mô hình với bối cảnh thể chế Việt Nam "
            "(Nghị quyết 57-NQ/TW; các QĐ 749/127/411/QĐ-TTg; cam kết COP26)."
        )
        for letter, question, answer in qa:
            st.markdown(f"**{letter}) {question}**")
            st.markdown(f"→ **Trả lời:** {answer}")
            st.markdown("")