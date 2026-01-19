
import torch
import numpy as np

def explain_prediction(ranker, tokenizer, resume_text, jd_text):
    """
    Generates an explanation for the ranking score using Attention weights.
    Returns: list of (token, score) tuples.
    """
    ranker.eval()
    inputs = tokenizer(
        resume_text, 
        jd_text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(ranker.device)
    
    with torch.no_grad():
        score, attentions = ranker.forward(
            inputs.input_ids, 
            inputs.attention_mask, 
            inputs.token_type_ids, 
            output_attentions=True
        )
        
    # Last layer, Average over Heads
    last_layer_attn = attentions[-1] 
    avg_head_attn = last_layer_attn.mean(dim=1) 
    cls_attention = avg_head_attn[0, 0, :] 
    
    input_ids = inputs.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    explanation = []
    for token, weight in zip(tokens, cls_attention):
        explanation.append((token, weight.item()))
        
    explanation.sort(key=lambda x: x[1], reverse=True)
    
    special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
    clean_explanation = [(t, w) for t, w in explanation if t not in special_tokens]
    
    return score.item(), clean_explanation[:20]

def compare_candidates(ranker, tokenizer, jd_text, resume_A, resume_B):
    """
    Computes Contrastive Attention: Contrast(t) = Attn_A(t) - Attn_B(t).
    High positive values indicate tokens that Candidate A covers better/more strongly than B.
    """
    # Helper to get attention map
    def get_attn(res_text):
        ranker.eval()
        inputs = tokenizer(res_text, jd_text, return_tensors="pt",  padding=True, truncation=True, max_length=512).to(ranker.device)
        with torch.no_grad():
            _, attentions = ranker.forward(inputs.input_ids, inputs.attention_mask, output_attentions=True)
        # Last layer, avg heads, CLS token attention
        cls_attn = attentions[-1].mean(dim=1)[0, 0, :]
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        return {t: w.item() for t, w in zip(tokens, cls_attn)}

    map_A = get_attn(resume_A)
    map_B = get_attn(resume_B)
    
    # Calculate contrast for shared tokens (intersection) or all tokens?
    # Focus on tokens present in the JD/Prompt usually, but here we look at tokens in the input stream.
    # We will iterate over tokens in Map A (A's strengths)
    
    contrast_scores = []
    for token, weight_A in map_A.items():
        if token in ["[CLS]", "[SEP]", "[PAD]"]: continue
        weight_B = map_B.get(token, 0.0)
        contrast = weight_A - weight_B
        contrast_scores.append((token, contrast))
        
    contrast_scores.sort(key=lambda x: x[1], reverse=True)
    return contrast_scores[:10] # Top 10 advantages of A over B

def format_explanation(explanation):
    result = "Top Influential Terms (Attention):\n"
    for token, weight in explanation:
        result += f"- {token}: {weight:.4f}\n"
    return result
