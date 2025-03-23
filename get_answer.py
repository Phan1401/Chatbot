from create_vocab import *
from processing import *
from load_data import answers
def get_answer(model,question, top_k=3, threshold=0.5):
    processed_q = preprocess_text(question)
    q_vector = numericalize(processed_q, word_vocab)
    q_vector = pad_sequences([q_vector], max_len)
    q_vector = torch.tensor(q_vector)

    # Tính cosine similarity
    similarities = cosine_similarity(q_vector.numpy(), numerical_questions.numpy())
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

    # Nếu có câu hỏi giống trên ngưỡng threshold, lấy câu trả lời từ dữ liệu
    best_idx = top_k_indices[0]
    if similarities[0][best_idx] >= threshold:
        best_answer = answers[best_idx]
        similar_questions = [questions[idx] for idx in top_k_indices]
    else:
        # Dùng mô hình Transformer để sinh câu trả lời
        with torch.no_grad():
            output = model(q_vector).squeeze(0)  # (max_len, vocab_size)
            predicted_indices = output.argmax(dim=1).tolist()
            best_answer = " ".join([word for word, idx in word_vocab.get_stoi().items() if idx in predicted_indices])

        # Dùng mô hình Transformer để tự sinh 3 câu hỏi tương tự
        with torch.no_grad():
            generated_questions = model.generate_text(q_vector, max_len=10)
            similar_questions = []
            for gen_q in generated_questions:
                text = " ".join([word for word, idx in word_vocab.get_stoi().items() if idx in gen_q.tolist()])
                similar_questions.append(text)

    return best_answer, similar_questions[:3]