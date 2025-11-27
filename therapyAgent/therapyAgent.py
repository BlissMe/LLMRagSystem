@router.post("/ask")
async def ask_question(data: QueryRequest):
    query = data.user_query
    history = data.history

    # MongoDB Setup
    client = MongoClient(key_param.MONGO_URI)
    db = client["Depression_Knowledge_Base"]
    collection = db["depression"]
    index_name = "default1"

    embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=index_name
    )
    similar_docs = vectorstore.similarity_search(query, k=3)
    context_texts = [doc.page_content[:500] for doc in similar_docs]
    summary_text = "\n".join(data.summaries) if data.summaries else "No previous summaries available."

    unasked_questions = [q for q in PHQ9_QUESTIONS if q["id"] not in data.asked_phq_ids]
    next_phq_q = unasked_questions[0] if unasked_questions else None

    # Determine early stage (first 2 user turns)
    user_turns = [line for line in data.history.splitlines() if line.lower().startswith("you:") or line.lower().startswith("user:")]
    early_stage = len(user_turns) < 3

    phq_instruction = ""
    if next_phq_q and not early_stage:
        if not data.asked_phq_ids:
            phq_instruction += f"""
You may now gently say something like:
"To better understand how you're doing, I'd like to ask a few short questions on how you feel in past two weeks."

Then ask this question:
- "{next_phq_q['question']}" (meaning: {next_phq_q['meaning']})
"""
        else:
            phq_instruction += f"""
Continue with the next question:
- "{next_phq_q['question']}" (meaning: {next_phq_q['meaning']})
"""
        phq_instruction += """
Make your response short and caring. Don't explain too much. No repetition. Only ask one PHQ-9 question per message.
Let user respond with:
- not at all
- several days
- more than half the days
- nearly every day
"""

    chat_prompt = f"""
You are a friendly chatbot who talks like a kind friend.

Be warm and caring. Avoid long or repetitive responses. Never say the same supportive line more than once.

Your job is to gently explore how the user feels and try to understand user by asking questions, and ask PHQ-9 questions naturally when ready.

NEVER mention PHQ-9 or say "I cannot help you".

Avoid medical or crisis terms unless directly asked.

Keep your replies short and friendly. One question per message. Once PHQ-9 starts, go through them without pausing.

Past summaries:
{summary_text}

Relevant context:
{context_texts}

Conversation history:
{history}

{phq_instruction}

User just said: "{query}"

Now reply like a kind friend:
"""

    bot = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=key_param.openai_api_key,
        temperature=0.7
    )

    chat_response = bot.invoke([{"role": "system", "content": chat_prompt}])
    final_text = chat_response.content.strip()
    client.close()

    matched_q = next_phq_q if not early_stage else None
    audio_path = generate_tts_audio(final_text)

    # ----------------------
    # PHQ-9 Progress
    # ----------------------
    total_phq9 = len(PHQ9_QUESTIONS)
    answered_phq9 = len(data.asked_phq_ids)
    phq9_progress = round((answered_phq9 / total_phq9) * 100, 2)
    phq9_started = bool(data.asked_phq_ids)
    phq9_completed = not unasked_questions

    # ----------------------
    # Log PHQ-9 Question Event
    # ----------------------
    try:
        monitor_payload = {
            "agent_name": "chat",
            "user_id": data.user_id,
            "session_id": data.session_id,
            "input_data": {
                "user_query": query,
                "history": history,
                "summaries": data.summaries,
                "asked_phq_ids": data.asked_phq_ids
            },
            "output_data": {
                "response": final_text,
                "phq9_questionID": matched_q["id"] if matched_q else None,
                "phq9_question": matched_q["question"] if matched_q else None,
                "phq9_started": phq9_started,
                "phq9_completed": phq9_completed,
                "phq9_progress": phq9_progress
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        requests.post("http://localhost:8000/monitor-agent/track-activity", json=monitor_payload, timeout=15)
        print("Logged PHQ-9 activity to Monitor Agent")
    except Exception as e:
        print("Failed to send PHQ-9 log to Monitor Agent:", e)

    # ----------------------
    # Log Follow-Up Chat Event if PHQ-9 Completed
    # ----------------------
    if phq9_completed:
        try:
            followup_payload = {
                "agent_name": "chat",
                "user_id": data.user_id,
                "session_id": data.session_id,
                "event": "FOLLOWUP_CHAT",
                "input_data": {
                    "history": history,
                    "user_query": query
                },
                "output_data": {
                    "response": final_text
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            requests.post("http://localhost:8000/monitor-agent/track-activity", json=followup_payload, timeout=15)
            print("Logged follow-up chat to Monitor Agent")
        except Exception as e:
            print("Failed to send follow-up chat log to Monitor Agent:", e)

    return {
        "response": final_text,
        "audio_url": f"/voice-audio?path={audio_path}",
        "phq9_questionID": matched_q["id"] if matched_q else None,
        "phq9_question": matched_q["question"] if matched_q else None,
        "phq9_progress": phq9_progress,
        "language": "English"
    }
