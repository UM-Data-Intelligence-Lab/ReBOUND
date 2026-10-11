import numpy as np

def sampler(data, cfg, gap_start, gap_end):
    counts = data['view_counts'].copy()
    table = data['views_by_train_day'].copy()
    length = data['view_windows'][:, 1] - data['view_windows'][:, 0] + 1
    calendar = data['calendar'][data['train_days']]
    enabled = calendar < gap_start if cfg['longest_segment'] == 'left' else calendar > gap_end
    for j in np.flatnonzero(enabled):
        vs = table[j, :counts[j]]
        chosen = vs[length[vs] == length[vs].max()]
        counts[j] = len(chosen)
        table[j, :len(chosen)] = chosen
    if 'position_threshold' not in cfg:
        return lambda ch, u: table[ch, (u * counts[ch]).astype(np.int64)]
    groups = np.full_like(table, -1)
    ranks = np.full_like(table, -1)
    allowed = []
    sizes = []
    penabled = calendar < gap_start if cfg['position_segment'] == 'left' else calendar > gap_end
    for (j, day) in enumerate(data['train_days']):
        vs = table[j, :counts[j]]
        w = data['view_windows'][vs]
        dist = np.minimum(day - w[:, 0] + 1, w[:, 1] - day + 1)
        for n in np.unique(length[vs]):
            ix = np.flatnonzero(length[vs] == n)
            keep = ix[dist[ix] <= cfg['position_threshold']] if penabled[j] else ix
            if not len(keep):
                keep = ix
            gid = len(allowed)
            groups[j, ix] = gid
            ranks[j, ix] = np.arange(len(ix))
            allowed.append(vs[keep])
            sizes.append(len(ix))
    sizes = np.array(sizes)
    ns = np.array([len(v) for v in allowed])
    pool = np.full((len(allowed), int(ns.max())), -1, dtype=table.dtype)
    for (i, v) in enumerate(allowed):
        pool[i, :len(v)] = v
    ratio = ns / sizes

    def choose(ch, u):
        q = u * counts[ch]
        offset = q.astype(np.int64)
        group = groups[ch, offset]
        phase = ranks[ch, offset] + (q - offset)
        pick = (phase * ratio[group]).astype(np.int64)
        assert np.all(pick < ns[group])
        return pool[group, pick]
    return choose
