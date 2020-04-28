
def decode_token_spans(spans_tokens, passage_text, question_text):
    spans_text = []
    spans_indices = []

    for context, tokens in spans_tokens:
        text_start = tokens[0].idx
        text_end = tokens[-1].idx + len(tokens[-1].lemma_)

        spans_indices.append((context, text_start, text_end))

        if context == 'p':
            spans_text.append(passage_text[text_start:text_end])
        else:
            spans_text.append(question_text[text_start:text_end])

    return spans_text, spans_indices

def get_token_context(token):
    if token.type_id == 0:
        context = 'q'
    elif token.type_id == 1:
        context = 'p'
    else:
        context = ''
    return context