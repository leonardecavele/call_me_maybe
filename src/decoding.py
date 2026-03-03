from llm_sdk import Small_LLM_Model


EOS: int = 151645
MAX_TOKENS: int = 256


def get_answers(model: Small_LLM_Model, prompts: list[str]) -> list[str]:
    prompts = prompts[:1] # to change

    answers: list[str] = []
    prompts_ids: list[list[int]] = [
        model.encode(p)[0].tolist() for p in prompts
    ]

    for prompt_ids in prompts_ids:
        input_ids: list[int] = prompt_ids[:]
        answer_ids: list[int] = []

        for _ in range(MAX_TOKENS):
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            next_id: int = max(range(len(logits)), key=logits.__getitem__)

            input_ids.append(next_id)
            if next_id == EOS:
                break
            answer_ids.append(next_id)

        answers.append(model.decode(answer_ids))

    return answers
