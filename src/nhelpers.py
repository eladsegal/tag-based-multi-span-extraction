import torch
import numpy as np
import string
from word2number.w2n import word_to_num
from allennlp.nn.util import replace_masked_values
from itertools import permutations

def tokenlist_to_passage(token_text):
    str_list = list(map(lambda x : x[2:] if len(x)>2 and x[:2]=="##" else " " + x, token_text))
    string = "".join(str_list)
    if string[0] == " ":
        string = string[1:]
    return string

def get_number_from_word(word):
    punctruations = string.punctuation.replace('-', '')
    word = word.strip(punctruations)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                number = None
    return number

def get_mask(mask_templates, numbers, ops):
    """get mask for next token"""
    with torch.no_grad():
        outmasks = torch.zeros((numbers.shape[0], numbers.shape[1], mask_templates.shape[-1]), device=numbers.device)
        mask_indices = (numbers > ops + 1).long().unsqueeze(-1).expand_as(outmasks)
        return torch.gather(mask_templates, 1, mask_indices, out=outmasks)

import pickle
def beam_search(K, log_probs, number_mask, op_mask, END, NUM_OPS):
    """beam search algorithm"""
    with torch.no_grad():
        # log_probs : (batch,seqlen,vocab)
        # number_mask : (batch,vocab)
        # op_mask : (batch,vocab)
        (batch_size, maxlen, V) = log_probs.shape
        
        # possible masks
        # mask_templates[0] : #nums = #ops + 1
        # mask_templates[1] : #nums > #ops + 1
        mask_templates = torch.zeros((batch_size, 2,V), device=log_probs.device)
        mask_templates[number_mask.unsqueeze(1).expand_as(mask_templates).byte()] = 1
        mask_templates[:,0,:][op_mask.byte()] = 0
        mask_templates[:,1,END] = 0
        mask_templates[:,0,END] = 1
        
        # expanded log_probs (for convinience)
        # log_probs2 : (batch,seqlen,K,vocab)
        log_probs2 = log_probs.unsqueeze(2).expand(-1,-1,K,-1)
        
        # #numbers so far
        numbers = torch.zeros((batch_size, K),device=log_probs.device).int()
        # #ops so far
        ops = torch.zeros((batch_size, K), device=log_probs.device).int()
              
        # best sequences and scores so far
        best_seqs = [[-100]] * batch_size
        best_scores = [-np.inf] * batch_size
        
        # initial mask
        init_mask = number_mask.clone()
        
        # first term
        scores = replace_masked_values(log_probs[:,0], init_mask, -np.inf)
        kscores, kidxs = scores.topk(K, dim=-1, largest=True, sorted=True)
          
        # update numbers and ops
        numbers += 1
        
        # K hypothesis for each batch
        # (batch, K, seqlen)
        seqs = kidxs.unsqueeze(-1)
        
        for t in range(1,maxlen):
            mask = get_mask(mask_templates, numbers, ops)
            scores = replace_masked_values(log_probs2[:,t], mask, -np.inf)
            tscores = (scores + kscores.unsqueeze(-1)).view(batch_size, -1)
            kscores, kidxs = tscores.topk(K, dim=-1, largest=True, sorted=True)
            # prev_hyps : (batch,K)
            prev_hyps = kidxs / V
            
            # next_tokens : (batch,K,1)
            next_tokens = (kidxs % V).unsqueeze(-1)
            
            if prev_hyps.max() >= K:
                print("problem")
                prev_hyps = torch.clamp(prev_hyps, max = K -1, min=0)
                
            # check how many have ended
            ended = next_tokens == END
            # update best_seqs and scores as needed
            for batch in range(batch_size):
                if ended[batch].sum() > 0:
                    ends = ended[batch].nonzero()
                    idx = ends[0,0]
                    token = next_tokens[batch, idx].cpu().item()
                    score = kscores[batch, idx].cpu().item()
                    if score > best_scores[batch]:
                        best_seqs[batch] = seqs[batch, prev_hyps[batch, idx]]
                        best_scores[batch] = score
                    for end in ends:
                        kscores[batch, end[0]] = -np.inf
            
            # update numbers and ops
            numbers = torch.gather(numbers, 1, prev_hyps)
            ops = torch.gather(ops, 1, prev_hyps)
            is_num = (next_tokens.squeeze() >= NUM_OPS).int()
            numbers += is_num.int()
            ops += (1 - is_num.int())
            
            # update seqs
            new_seqs = torch.gather(seqs, 1, prev_hyps.unsqueeze(-1).expand_as(seqs))
            seqs = torch.cat([new_seqs, next_tokens], -1)
#             print(seqs)
#         with open("output.txt", "a") as myfile:
#             print("best_seqs : ", best_seqs, file=myfile)
#             print("seqs : ", seqs, file=myfile)
        return best_seqs, best_scores

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def evaluate_postfix(exp):
    with torch.no_grad():
        stack = []
        for t in exp:
            if is_number(t):
                stack.append(t)
            else:
                if len(stack) > 1:
                    val1 = stack.pop()
                    val2 = stack.pop()
                    stack.append(str(eval(val2 + t + val1)))
                else:
                    print("bad")
                    break
        try:
            result = stack.pop()
        except:
            result = "0"
        return result
    
def get_full_exp(numbers, targets, operations, op_dict, max_depth):
    num_ops = len(operations)
    expressions = []
    for depth in range(2, max_depth + 1):
        stack = [([], set(), 0, 0, [])]
        while stack:
            exp, used_nums, num_num, num_op, eval_stack = stack.pop()
            # Expression complete
            if len(exp) == 2 * depth - 1:
                if eval_stack[0] in targets:
                    expressions.append(exp + [(0, '')])

            # Can add num
            if num_num < depth:
                for num in numbers:
                    if num not in used_nums:
                        new_exp = exp + [(num[0] + num_ops + 1, num[1])]
                        new_used_nums = used_nums.copy()
                        new_used_nums.add(num)
                        new_eval_stack = eval_stack + [num[1]]
                        stack.append((new_exp, new_used_nums, num_num + 1, num_op, new_eval_stack))

            # Can add op
            if num_op < depth - 1 and len(eval_stack) >= 2:
                for op in operations:
                    try:
                        result = op_dict[op[1]](eval_stack[-2], eval_stack[-1])
                        new_exp = exp + [(op[0] + 1, op[1])]
                        new_eval_stack = eval_stack[:-2] + [result]
                        stack.append((new_exp, used_nums, num_num, num_op + 1, new_eval_stack))
                    except ZeroDivisionError:
                        pass
        
    for ind, exp in enumerate(expressions):
        zipped = list(zip(*exp))
        expressions[ind] = (list(zipped[0]), ' '.join([str(x) for x in zipped[1]]))
    return expressions

def clipped_passage_num(number_indices, number_len, numbers_in_passage, plen):
    if number_indices[-1] < plen:
        return number_indices, number_len, numbers_in_passage
    lo = 0
    hi = len(number_indices) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if number_indices[mid] < plen:
            lo = mid + 1
        else:
            hi = mid
    if number_indices[lo - 1] + number_len[lo - 1] > plen:
        number_len[lo - 1] = plen - number_indices[lo - 1]
    return number_indices[:lo], number_len[:lo], numbers_in_passage[:lo]

def get_answer_type(answers):
    if answers['number']:
        return 'number'
    elif answers['spans']:
        if len(answers['spans']) == 1:
            return 'single_span'
        return 'multiple_span'
    elif any(answers['date'].values()):
        return 'date'
    
def get_template_exp(numbers, targets, templates, template_strings):
    valid_expressions_indices = [[] for _ in range(len(templates))]
    valid_expressions_strings = [[] for _ in range(len(templates))]
    for number_combination in permutations(enumerate(numbers), 3):
        num_indices, num = list(zip(*number_combination))
        num_strings = tuple([str(x) for x in num])
        for ind, template in enumerate(templates):
            try:
                if template(*num) in targets:
                    valid_expressions_indices[ind].append(list(num_indices))
                    valid_expressions_strings[ind].append(template_strings[ind] % num_strings)
            except ZeroDivisionError:
                continue 
    return valid_expressions_indices, valid_expressions_strings
   