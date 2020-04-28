
def tokenize_with_offsets(tokenize_func, text,
                          **tokenize_func_kwargs):
    """
    Keep track of token offsets by trying to progressively tokenize the text character by character, 
    and consume matching tokens along the way.
    For efficiency, parts of the text that were already matched should be discarded.
    However, since some tokenizations are dependent on neighboring past text,
    in cases of mismatch we fall back to use previously matched parts of the text.
    For some tokenizers, more than a single token can be generated for a single character.
    In such cases, the nonfinal tokens will have the same offset as the final token.
    """

    # Tokenize text
    tokens = tokenize_func(text, **tokenize_func_kwargs)

    # Get maximum search length
    max_word_length = max([len(word) for word in text.split()])
    max_space_length = _get_max_space_length(text)
    max_search_length = max_word_length + max_space_length

    # Initialize token iteration variables
    boundary_token_index = 0
    prev_boundary_token_indexes = [0]
    token_offsets = [0]
    i = 0
    retry = 0
    while i < len(tokens):
        token = tokens[i]
        match_error = False

        if retry > 0:
            # The tokenization from the boundary doesn't match the text, retrying with a previous boundary
            boundary_token_index = prev_boundary_token_indexes[-retry]
        else:
            # Try boundary of the current token
            boundary_token_index = i

        # Initialize search variables
        offset = token_offsets[i]
        search_length = 0
        comparison_tokens = []
        while True:
            prev_comparison_tokens = comparison_tokens # for debugging
            comparison_tokens = _get_comparison_tokens(tokenize_func, text, 
                                                        token_offsets[boundary_token_index], offset, search_length, 
                                                        **tokenize_func_kwargs)

            target_tokens = tokens[boundary_token_index : i + 1]
            if _is_prefix(target_tokens, comparison_tokens):
                # Found a tokenization match
                if len(comparison_tokens) > len(target_tokens):
                    # Handle special cases
                    index = text[offset : offset + search_length].find(token)
                    if (index != -1):
                        # Words that have a wordpiece tokenization that 
                        # doesn't contain the tokenization of its prefixes
                        search_length = index + len(token)
                    else:
                        overreach_length = len(comparison_tokens) - len(target_tokens)
                        overreach_tokens = tokens[i + 1 : i + 1 + overreach_length]
                        if comparison_tokens[len(target_tokens) : ] != overreach_tokens:
                            # For cases in which the current token won't be produced 
                            # without an additional character that is only part of the
                            # text that corresponds to the next tokens.
                            # Example for XLNet: 
                            # How many points did the buccaneers need to tie in the first?
                            # tokens = [..., '▁the', '▁', 'bu', 'cca', 'ne', 'ers', ...]
                            # target_tokens = ['▁']
                            # comparison_tokens = ['▁', 'b']
                            # prev_comparison_tokens = ['']
                            search_length -= 1
                        else:
                            # Characters that are tokenized to two or more tokens
                            search_length = 0

                # Store successful boundary
                if prev_boundary_token_indexes[-1] != boundary_token_index:
                    prev_boundary_token_indexes.append(boundary_token_index)
                retry = 0
                break

            if search_length == max_search_length:
                # The tokenization from the boundary doesn't match the text, retry with a previous boundary,
                # keep retrying until all the previous successful boundaries are used
                match_error = True
                if retry < len(prev_boundary_token_indexes):
                    retry += 1
                else:
                    retry = 0
                break

            # Search step
            search_length += 1

        if match_error:
            if retry > 0:
                continue
            # Failed to match offsets to the tokens
            break

        # Keep consuming characters until there's no tokenization match.
        # Required due to special characters such as in "Moskva: Russkiĭ fond sodeĭstviii︠a︡ obrazovanii︠u︡ i nauke"
        if comparison_tokens == target_tokens:
            while True:
                if len(text) == offset + search_length:
                    break

                comparison_tokens = _get_comparison_tokens(tokenize_func, text, 
                                                            token_offsets[boundary_token_index], offset, search_length + 1, 
                                                            **tokenize_func_kwargs)
                if _is_prefix(comparison_tokens, target_tokens):
                    search_length += 1
                else:
                    break

        if len(text) != offset + search_length:
            # Add the next token offset only if the end of the text wasn't reached
            token_offsets.append(offset + search_length)
        else:
            break
        i += 1

    if match_error:
        raise Exception("Failed to tokenize with offsets: Unknown reason.")

    # Construct the original text that corresponds to each token, up to spaces
    original_token_texts = []
    for i, offset in enumerate(token_offsets):
        if i < len(token_offsets) - 1:
            next_offset = token_offsets[i + 1]
            original_token_text = text[offset:next_offset]
        else:
            original_token_text = text[offset:]
        original_token_texts.append(original_token_text) 

    for i, original_token_text in enumerate(original_token_texts):
        if len(original_token_text.strip().split()) > 1:
            raise Exception(f"Failed to tokenize with offsets: A token is over-consuming, probably due to a bad character in the input: #f{i} - {original_token_text}")

    return (tokens, token_offsets)

def align_tokens_to_tokens(from_tokens, to_tokens):
    """
    Based on fairseq's alignment utils:
    https://github.com/pytorch/fairseq/blob/8446cb6385b4f9aec422a469029ed9c900867955/fairseq/models/roberta/alignment_utils.py#L39
    
    Returns:
        List[int]: mapping from `from_tokens` to corresponding `to_tokens`:
                   The size of the list is the size of `from_tokens`, the values are values in `range(0, len(to_tokens))`
    """
    alignment = []

    to_iterator = enumerate(to_tokens)
    j, to_token = next(to_iterator)
    for from_token in from_tokens:
        indices = []
        while True:
            if from_token.startswith(to_token):
                indices.append(j)
                from_token = from_token[len(to_token):]
                j, to_token = next(to_iterator, (None, None))
            elif to_token.startswith(from_token):
                # from_token spans multiple to_tokens
                indices.append(j)
                to_token = to_token[len(from_token):]
                from_token = ""
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(from_token, to_token))
            if from_token == "":
                break
        assert len(indices) > 0
        alignment.append(indices)
    assert len(alignment) == len(from_tokens)

    return alignment

def alignment_to_wordpieces_list(alignment):
    """
    Returns a mapping between word indices and the wordpieces indices they were tokenized to.
    If the words has a split that the wordpieces don't have then we consider it as if there wasn't a split.
    For example, spaCy tokenizes "Im" to "I" and "m", and a wordpiece tokenizer would tokenize to "Im".
    In this case, wordpieces will be [[0]].

    Example usage:
    text = "officiating"
    tokens, offsets = tokenize_with_offsets(wordpiece_tokenize_func, text) # tokens = ["off", "##icia", "##ting"]

    word_tokenizer = SpacyTokenizer()
    words = [word.text for word in word_tokenizer.tokenize(text)] # words = ["officiating"]
    
    alignment = align_tokens_to_tokens(text, words, tokens)
    wordpieces_list = alignment_to_wordpieces_list(alignment) # [[0, 1, 2]]
    """
    wordpieces_list = []
    for indices in alignment:
        if wordpieces_list and wordpieces_list[-1][-1] == indices[0]:
            wordpieces_list[-1] += indices[1:]
        else:
            wordpieces_list.append(indices)
    return wordpieces_list

def wordpieces_list_to_token_wordpieces(wordpieces_list):
    token_wordpieces = []
    for wordpieces in wordpieces_list:
        for wordpiece in wordpieces:
            token_wordpieces.append(wordpieces)
    return token_wordpieces

def token_offsets_to_strings(offsets, text):
    strings = []
    for i, offset in enumerate(offsets):
        if i < len(offsets) - 1:
            next_offset = offsets[i + 1]
            string = text[offset : next_offset]
        else:
            string = text[offset : ]
        string = string.strip()
        strings.append(string)
    return strings

"""
Helper functions
"""
def _get_max_space_length(text):
    max_space_length = 0
    count = False
    for i, c in enumerate(text):
        if c == " ":
            if not count:
                count = True
                start_index = i
        else:
            if count:
                count = False
                max_space_length = max(max_space_length, i - start_index)
    return max_space_length

def _is_prefix(lst, other_lst):
    """
    Checks if `lst` is a prefix of `other_lst` 
    """
    if len(lst) > len(other_lst):
        return False
    return other_lst[:len(lst)] == lst

def _get_comparison_tokens(tokenize_func, text, boundary_token_offset, token_offset, 
                           search_length, **tokenize_func_kwargs):
    search_text = text[boundary_token_offset : token_offset + search_length]
    return tokenize_func(search_text, **tokenize_func_kwargs)
