import torch


def tensorize_triples_MC(query_tokenizer, doc_tokenizer, passage, target, opt1, opt2, opt3, opt4, bsize):
    assert len(passage) == len(target) == len(opt1) == len(opt2) == len(opt3) == len(opt4)
    assert bsize is None or len(passage) % bsize == 0

    N = len(passage)
    Q_ids, Q_mask = query_tokenizer.tensorize(passage)
    D_ids, D_mask = doc_tokenizer.tensorize(target + opt1 + opt2 + opt3 + opt4)
    D_ids, D_mask = D_ids.view(5, N, -1), D_mask.view(5, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (target_ids, opt1_ids, opt2_ids, opt3_ids, opt4_ids), (target_mask, opt1_mask, opt2_mask, opt3_mask, opt4_mask) = D_ids, D_mask

    passage_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    target_batches = _split_into_batches(target_ids, target_mask, bsize)
    opt1_batches = _split_into_batches(opt1_ids, opt1_mask, bsize)
    opt2_batches = _split_into_batches(opt2_ids, opt2_mask, bsize)
    opt3_batches = _split_into_batches(opt3_ids, opt3_mask, bsize)
    opt4_batches = _split_into_batches(opt4_ids, opt4_mask, bsize)

    batches = []
    for (p_ids, p_mask), (t_ids, t_mask), (o1_ids, o1_mask), (o2_ids, o2_mask), (o3_ids, o3_mask), (o4_ids, o4_mask) in zip(passage_batches, target_batches, opt1_batches, opt2_batches, opt3_batches, opt4_batches):
        Q = (torch.cat((p_ids, p_ids, p_ids, p_ids, p_ids)), torch.cat((p_mask, p_mask, p_mask, p_mask, p_mask)))
        D = (torch.cat((t_ids, o1_ids, o2_ids, o3_ids, o4_ids)), torch.cat((t_mask, o1_mask, o2_mask, o3_mask, o4_mask)))
        batches.append((Q, D))

    return batches

def tensorize_triples_MC_train(query_tokenizer, doc_tokenizer, query_t, query_1, query_2, query_3, query_4, positive, negative, bsize):
    assert len(query_t) == len(positive) == len(negative)
    assert bsize is None or len(query_t) % bsize == 0

    N = len(query_t)
    Qt_ids, Qt_mask = query_tokenizer.tensorize(query_t)
    Q1_ids, Q1_mask = query_tokenizer.tensorize(query_1)
    Q2_ids, Q2_mask = query_tokenizer.tensorize(query_2)
    Q3_ids, Q3_mask = query_tokenizer.tensorize(query_3)
    Q4_ids, Q4_mask = query_tokenizer.tensorize(query_4)
    D_ids, D_mask = doc_tokenizer.tensorize(positive + negative)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Qt_ids, Qt_mask = Qt_ids[indices], Qt_mask[indices]
    Q1_ids, Q1_mask = Q1_ids[indices], Q1_mask[indices]
    Q2_ids, Q2_mask = Q2_ids[indices], Q2_mask[indices]
    Q3_ids, Q3_mask = Q3_ids[indices], Q3_mask[indices]
    Q4_ids, Q4_mask = Q4_ids[indices], Q4_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (pos_ids, neg_ids), (pos_mask, neg_mask) = D_ids, D_mask

    query_batches_t = _split_into_batches(Qt_ids, Qt_mask, bsize)
    query_batches_1 = _split_into_batches(Q1_ids, Q1_mask, bsize)
    query_batches_2 = _split_into_batches(Q2_ids, Q2_mask, bsize)
    query_batches_3 = _split_into_batches(Q3_ids, Q3_mask, bsize)
    query_batches_4 = _split_into_batches(Q4_ids, Q4_mask, bsize)
    positive_batches = _split_into_batches(pos_ids, pos_mask, bsize)
    negative_batches = _split_into_batches(neg_ids, neg_mask, bsize)

    batches = []
    for (qt_ids, qt_mask), (q1_ids, q1_mask), (q2_ids, q2_mask), (q3_ids, q3_mask), (q4_ids, q4_mask), \
        (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches_t, query_batches_1, query_batches_2, 
                                                query_batches_3, query_batches_4,
                                                positive_batches, negative_batches):
        Q = (torch.cat((qt_ids, qt_ids, q1_ids, q1_ids, q2_ids, q2_ids, q3_ids, q3_ids, q4_ids, q4_ids)), 
             torch.cat((qt_mask, qt_mask, q1_mask, q1_mask, q2_mask, q2_mask, q3_mask, q3_mask, q4_mask, q4_mask)))
        D = (torch.cat((p_ids, n_ids, p_ids, n_ids, p_ids, n_ids, p_ids, n_ids, p_ids, n_ids)), 
             torch.cat((p_mask, n_mask, p_mask, n_mask, p_mask, n_mask, p_mask, n_mask, p_mask, n_mask)))
        batches.append((Q, D))

    return batches


def tensorize_triples_MC_test(query_tokenizer, doc_tokenizer, query_t, query_1, query_2, query_3, query_4, passage, bsize):
    assert len(query_t) == len(passage)
    assert bsize is None or len(query_t) % bsize == 0

    N = len(query_t)
    Qt_ids, Qt_mask = query_tokenizer.tensorize(query_t)
    Q1_ids, Q1_mask = query_tokenizer.tensorize(query_1)
    Q2_ids, Q2_mask = query_tokenizer.tensorize(query_2)
    Q3_ids, Q3_mask = query_tokenizer.tensorize(query_3)
    Q4_ids, Q4_mask = query_tokenizer.tensorize(query_4)
    D_ids, D_mask = doc_tokenizer.tensorize(passage + ["place holder"]*len(passage))
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Qt_ids, Qt_mask = Qt_ids[indices], Qt_mask[indices]
    Q1_ids, Q1_mask = Q1_ids[indices], Q1_mask[indices]
    Q2_ids, Q2_mask = Q2_ids[indices], Q2_mask[indices]
    Q3_ids, Q3_mask = Q3_ids[indices], Q3_mask[indices]
    Q4_ids, Q4_mask = Q4_ids[indices], Q4_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (pas_ids,_), (pas_mask,_) = D_ids, D_mask

    query_batches_t = _split_into_batches(Qt_ids, Qt_mask, bsize)
    query_batches_1 = _split_into_batches(Q1_ids, Q1_mask, bsize)
    query_batches_2 = _split_into_batches(Q2_ids, Q2_mask, bsize)
    query_batches_3 = _split_into_batches(Q3_ids, Q3_mask, bsize)
    query_batches_4 = _split_into_batches(Q4_ids, Q4_mask, bsize)
    passage_batches = _split_into_batches(pas_ids, pas_mask, bsize)

    batches = []
    for (qt_ids, qt_mask), (q1_ids, q1_mask), (q2_ids, q2_mask), (q3_ids, q3_mask), (q4_ids, q4_mask), \
        (p_ids, p_mask) in zip(query_batches_t, query_batches_1, query_batches_2, 
                                                query_batches_3, query_batches_4,
                                                passage_batches):
        Q = (torch.cat((qt_ids, q1_ids, q2_ids, q3_ids, q4_ids)), 
             torch.cat((qt_mask, q1_mask, q2_mask, q3_mask, q4_mask)))
        D = (torch.cat((p_ids, p_ids, p_ids, p_ids, p_ids)), 
             torch.cat((p_mask, p_mask, p_mask, p_mask, p_mask)))
        batches.append((Q, D))

    return batches

    # assert len(passage) == len(target) == len(opt1) == len(opt2) == len(opt3) == len(opt4)
    # assert bsize is None or len(passage) % bsize == 0

    # N = len(target)
    # Q_ids_t, Q_mask_t = query_tokenizer.tensorize(target)
    # Q_ids_o1, Q_mask_o1 = query_tokenizer.tensorize(opt1)
    # Q_ids_o2, Q_mask_o2 = query_tokenizer.tensorize(opt2)
    # Q_ids_o3, Q_mask_o3 = query_tokenizer.tensorize(opt3)
    # Q_ids_o4, Q_mask_o4 = query_tokenizer.tensorize(opt4)
    # D_ids, D_mask = doc_tokenizer.tensorize(passage)
    # D_ids, D_mask = D_ids.view(1, N, -1), D_mask.view(1, N, -1)

    # # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    # maxlens = D_mask.sum(-1).max(0).values

    # # Sort by maxlens
    # indices = maxlens.sort().indices
    # Q_ids_t, Q_mask_t = Q_ids_t[indices], Q_mask_t[indices]
    # Q_ids_o1, Q_mask_o1 = Q_ids_o1[indices], Q_mask_o1[indices]
    # Q_ids_o2, Q_mask_o2 = Q_ids_o2[indices], Q_mask_o2[indices]
    # Q_ids_o3, Q_mask_o3 = Q_ids_o3[indices], Q_mask_o3[indices]
    # Q_ids_o4, Q_mask_o4 = Q_ids_o4[indices], Q_mask_o4[indices]
    # D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    # (passage_ids), (passage_mask) = D_ids, D_mask

    # passage_batches = _split_into_batches(passage_ids, passage_mask, bsize)
    # target_batches = _split_into_batches(Q_ids_t, Q_ids_t, bsize)
    # opt1_batches = _split_into_batches(Q_ids_o1, Q_mask_o1, bsize)
    # opt2_batches = _split_into_batches(Q_ids_o2, Q_mask_o2, bsize)
    # opt3_batches = _split_into_batches(Q_ids_o3, Q_mask_o3, bsize)
    # opt4_batches = _split_into_batches(Q_ids_o4, Q_mask_o4, bsize)

    # batches = []
    # for (p_ids, p_mask), (t_ids, t_mask), (o1_ids, o1_mask), (o2_ids, o2_mask), (o3_ids, o3_mask), (o4_ids, o4_mask) in zip(passage_batches, target_batches, opt1_batches, opt2_batches, opt3_batches, opt4_batches):
    #     Q = (torch.cat((t_ids, o1_ids, o2_ids, o3_ids, o4_ids)), torch.cat((t_mask, o1_mask, o2_mask, o3_mask, o4_mask)))
    #     D = (torch.cat((p_ids, p_ids, p_ids, p_ids, p_ids)), torch.cat((p_mask, p_mask, p_mask, p_mask, p_mask)))
    #     batches.append((Q, D))

    # return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
