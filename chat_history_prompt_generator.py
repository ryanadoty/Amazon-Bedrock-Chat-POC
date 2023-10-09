question_history = []


def chat_history(session_state):
    question = ""
    answer = ""
    question_answer = dict()

    for message in session_state.get('messages'):
        # print(message)
        if message.get('role') == 'user':
            question = message.get('content')
        if message.get('role') == 'assistant':
            answer = message.get('content')

        question_answer = {
            "question": question,
            "answer": answer
        }
    question_history.append(question_answer)

    if len(question_history) > 4:
        question_history.remove(question_history[0])
    else:
        pass

    final_prompt = ""
    for question_answer_pair in question_history:
        # print(question_answer_pair)
        final_prompt += f"""
        
        Question: {question_answer_pair.get('question')}
        
        Answer: {question_answer_pair.get('answer')}"""
        # print("-------------------------------------------")
        # print(final_prompt)
        print("-------------------------------------------")
    # print(final_prompt)
    with open('chat_history.txt', 'w') as history:
        history.write(final_prompt)
    return final_prompt
