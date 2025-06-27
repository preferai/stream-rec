
### 1. HOMETOWN
**Request**
```json
{
  "user_id": "u123"
}
```

**Response**
```json
{
  "streams": [
    { "stream_id": "s11", "city": "Zagreb" },
    { "stream_id": "s14", "city": "Zagreb" }
  ]
}
```

**Related Work**  
Zheng, Y., Wilkie, D., & Mokbel, M. F. (2015). Recommendations in location-based social networks: A survey. *GeoInformatica, 19*(3), 525–565. https://doi.org/10.1007/s10707-014-0220-8

**Implementation Insight**  
Using the **`user_id`** we fetch the user’s city/geo-hash; we then match it against each stream’s geo tag to boost local candidates. Using the paper’s geo‑social scoring, we add a proximity boost B = 1 / (1 + distance_km) to each live stream whose city tag matches. The reranker sorts by (original_score + B) so local channels surface first.

**Data used for training:** user locations (geo-hash/city), streamer locations, historical user–stream location matches.


---

### 2. SMS_ALERT
**Request** none

**Response (generated batch)**
```json
{
  "user_id": "u123",
  "stream_id": "fav77",
  "sms_copy": "We miss you! Your favourite streamer is live – jump back in."
}
```

**Related Work**  
Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). *DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network.* **BMC Medical Research Methodology, 18**, 24. https://doi.org/10.1186/s12874-018-0482-1

**Implementation Insight**  
For each **`user_id`** we build feature vectors (recency, frequency, total watch-minutes, platform, etc.) and feed them to **DeepSurv**, a neural Cox model. It outputs log-risk → survival *S(t)*. If *S(t) < 0.4* and **last_login > 30 days**, we queue the user and inject their top-watched `stream_id` into the SMS—mirroring DeepSurv’s hazard-based intervention logic.

**Data used for training:** user visit timestamps, last-login gap, streams bookmarked or followed by the user, historical SMS send/open labels.

---

### 3. HOME_FEED
**Request**
```json
{
  "user_id": "u123",
  "ctx_timestamp": 1718803200,
  "ctx_device": "ios",
  "ctx_locale": "en_US"
}
```

**Response**
```json
{
  "request_id": "req-001",
  "streams": [
    { "stream_id": "s77", "score": 0.93, "reason": "Based on your history" }
  ]
}
```

**Related Work**  
He, X. et al. (2020). *LightGCN: Simplifying and powering graph convolution network for recommendation.* **SIGIR 2020**, 639–648. https://arxiv.org/abs/2002.02126

**Implementation Insight**  
LightGCN embeddings for **`user_id`** and each `stream_id` give dot-product scores. We then add context tweaks: for example +3 % if `ctx_device` matches the streamer’s dominant device cluster, −2 % if `ctx_locale` differs from stream language (infer those values from streamers distribution)

**Data used for training:** full user-stream interaction logs (views, follows, watch-time), timestamps, device type, locale, stream metadata (category, language, viewer count).

---

### 4. CATEGORY_FEED
**Request**
```json
{
  "user_id": "u123",
  "category_id": "valorant",
  "ctx_locale": "en_US"
}
```

**Response**
```json
{
  "request_id": "req-002",
  "streams": [
    { "stream_id": "s55", "score": 0.87, "reason": "Popular in Valorant" }
  ]
}
```

**Related Work**  
Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* **EMNLP-IJCNLP 2019**, 3982–3992. https://aclanthology.org/D19-1410/

**Implementation Insight**  
We embed `category_id` text and each stream title/tag with Sentence-BERT (**cosine similarity S** per §3). Final score = **α·S + (1−α)·CF** where CF is LightGCN; we start α = 0.7 and tune via offline NDCG for example.

**Data used for training:** user interaction counts per category, stream metadata (category, tags, title text), user locale, stream popularity signals.



---

### 5. UP_NEXT
**Request**
```json
{
  "user_id": "u123",
  "current_stream_id": "s77",
  "watch_duration": 1420
}
```

**Response**
```json
{
  "next": [
    { "stream_id": "s88", "expected_watch_sec": 900 },
    { "stream_id": "s91", "expected_watch_sec": 600 }
  ]
}
```


**Related Work**  
Kang, W.-C., & McAuley, J. (2018). *Self-attentive sequential recommendation.* **ICDM 2018**, 197–206. https://arxiv.org/abs/1808.09781

**Implementation Insight**  
We build a 50-item sequence (pad if needed) with **`current_stream_id`** appended, embed each ID, add sinusoidal positions, and feed it to SASRec. Following the paper’s idea that more recent and engaging items should receive higher attention, we multiply every key/query vector by `(watch_duration / 1800)` and clip the factor to 0.5–2.0. Thus a 30-min view stays at 1.0, longer views up-weight to 2×, and quick exits down-weight to 0.5×. This way, long-watched streams speak louder in attention. SASRec outputs next-item probabilities and a regression head yields `expected_watch_sec` for the top-k results.


**Data used for training:** chronological user watch sequences with per-item watch duration, stream embeddings, timestamps, device context.

---

### 6. SEARCH_SUGGEST
**Request**
```json
{
  "user_id": "u123",
  "query_text": "speedrun"
}
```

**Response**
```json
{
  "results": [
    { "stream_id": "s55", "score": 0.81, "highlight": "speed<span>run</span>" },
    { "stream_id": "s81", "score": 0.73, "highlight": "Speedrun World Record" }
  ]
}
```

**Related Work**  
Huang, P.-S. et al. (2013). *Learning deep structured semantic models for web search using click-through data.* **CIKM 2013**, 2333–2338. https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/

**Implementation Insight**  
`query_text` is normalised and encoded via DSSM into vector **q**; each stream’s vector **d** (title, tags, game) is cached in a Faiss ANN index. We retrieve the top 500 by cosine(q,d) and rescore with cosine × personalised click-prior **P<sub>click</sub>(user_id, stream_id)**, which is learned with a logistic model on historical query‑click pairs (like the paper’s click‑through loss). This mirrors the click-through training objective and couples semantic relevance with behavioural likelihood.

 **Data used for training:** historical query-click pairs, query text n-grams, stream title/tags, user click-through history, click labels.

---

### 7. FOLLOW_RECOMMEND
**Request**
```json
{
  "user_id": "u123"
}
```

**Response**
```json
{
  "creators": [
    { "stream_id": "c44", "reason": "Similar audience" },
    { "stream_id": "c18", "reason": "Watched by users like you" }
  ]
}
```
**Related Work**  
He, X. et al. (2020). *LightGCN* (see Scenario 3). https://arxiv.org/abs/2002.02126

**Implementation Insight**  
Follower graph embeddings from LightGCN yield creator similarities. We exclude already-followed creators for **`user_id`** and rank remaining ones by similarity plus social-proof (common followers). 

**Data used for training:** user-creator follow graph, mutual-follower counts, creator metadata (categories, average viewers), user viewing history.

---

### 8. CLIP_DIGEST
**Response (generated batch)**
```json
{
  "user_id": "u123",
  "clips": [
    { "clip_id": "cl1", "title": "Epic clutch!", "thumb": "https://..." },
    { "clip_id": "cl2", "title": "Insane play", "thumb": "https://..." }
  ]
}
```

**Related Work**  
Qin, M., Karatzoglou, A., & Baltrunas, L. (2019). *Personalized re-ranking for recommendation.* arXiv:1904.06813. https://arxiv.org/abs/1904.06813

**Implementation Insight**  
A scheduled job gathers all clips published in the last 24 h that **`user_id`** has not watched. Each clip is embedded with SBERT, while the user profile vector is the watch‑time‑weighted average of embeddings from their recent viewing history. We compute four features per clip: (1) semantic similarity S = cosine(user, clip), (2) logarithmic popularity, (3) recency in hours, and (4) a flag for "creator followed by user". These features are fed into LambdaMART trained with pairwise loss on past digest click‑through (per Qin et al. §3). LambdaMART scores are then multiplied by an exponential freshness decay to promote new clips, the top‑N are selected, and the digest payload is formatted as above.

**Data used for training:** clip embeddings (title, tags), clip publish time, clip popularity (views, likes), user watch-time-weighted embeddings, user follow list, past digest click labels.

---

### 9. RAID_SUGGEST
**Request**
```json
{
  "current_stream_id": "s77"
}
```

**Response**
```json
{
  "targets": [
    { "stream_id": "s90", "overlap_score": 0.72 },
    { "stream_id": "s12", "overlap_score": 0.65 }
  ]
}
```

**Related Work**  
Velicković, P. et al. (2018). *Graph Attention Networks.* **ICLR 2018**. https://arxiv.org/abs/1710.10903

**Implementation Insight**  
Build a creator graph whose edges store **shared-viewer fraction** (percentage of current viewers who have watched both creators) and **post-raid retention** (fraction of raiders who stay ≥ 5 min).  These become edge features; node features include category mix and average watch-time.  A multi-head **Graph Attention Network** is trained to predict next-raid retention, so the learnt embeddings encode both audience overlap *and* how well raids “stick.”
At runtime we seed the network with **`current_stream_id`** and take the cosine similarity between its embedding and every candidate’s embedding.  That similarity is multiplied by an **expected-retention** value predicted by a MLP (input = overlap, category match, candidate size), giving  `synergy_score = similarity × expected_retention` – a partner must be topically close *and* good at keeping raiders.
To stop huge channels dominating, we divide `synergy_score` by `(impressions + 1)^β` (β = 0.3 for example) where *impressions* is the last-7-day raid-carousel exposure for that candidate (i.e., a count).  This inverse-exposure adjustment lets smaller high-synergy creators outrank bigger ones when their base scores are comparable.
  
**Data used for training:** viewer-overlap ratios between channels, historical raid events with retention outcomes, creator categories, channel size metrics, raid-carousel impression logs.

---

### 10. LIVE_NOTIFY
**Response (push event)**
```json
{
  "user_id": "u123",
  "stream_id": "s77",
  "message": "🚨 xQc just went live! Tap to watch now."
}
```

**Related Work**  
McMahan, H. B. et al. (2013). *Ad click prediction.* **KDD 2013**, 1222-1230. <https://research.google/pubs/pub41159/>

**Implementation Insight**  
Logistic regression model to predict CTR on follow-age, time-bucket, `ctx_device`, creator popularity, prior open history; send push if probability > 0.25, copy templated by an LLM.

**Data used for training:** push notification history (sent/opened), user follow list, creator live-start timestamps, device type, time-of-day buckets.

---

### 11. DISCOVERY_ROW
**Request**
```json
{
  "user_id": "u123",
  "anchor_stream_id": "s77"
}
```

**Response**
```json
{
  "streams": [
    { "stream_id": "s91", "score": 0.66 },
    { "stream_id": "s10", "score": 0.61 }
  ]
}
```

**Related Work**  
Gomez-Uribe, C. A., & Hunt, N. (2015). *The Netflix recommender system.* **ACM TMIS, 6**(4), 13. <https://dl.acm.org/doi/abs/10.1145/2843948>

**Implementation Insight**  
Hybrid of LightGCN item-item similarity and SBERT text similarity (50/50) ranks 20 “Because you watched” streams.

**Data used for training:** user watch history, pairwise stream similarity scores (LightGCN), stream metadata, co-watch statistics.

---

### 12. COLD_CREATOR_BOOST
**Request**
```json
{
  "user_id": "u123",
  "ctx_locale": "en_US"
}
```

**Response**
```json
{
  "streams": [
    { "stream_id": "s99", "fairness_factor": 1.8 },
    { "stream_id": "s100", "fairness_factor": 1.7 }
  ]
}
```

**Related Work**  
Mansoury, M., & Mobasher, B. (2023). *Fairness of exposure in dynamic recommendations.* <https://arxiv.org/abs/2309.02322>

**Implementation Insight**  
Inspired by the paper’s exposure-aware framework (§3), we could maintain a 30-day rolling cumulative exposure counter for every creator. At ranking time we start with a raw relevance score (e.g., LightGCN) and blend it with inverse exposure using a tunable weight λ:  
`score_final = (1 − λ) × relevance + λ × (1 / (cumExposure + 1))`, with **λ = 0.25** tuned offline. This mirrors the proportional-exposure interpolation in the paper, encouraging under-served creators without discarding relevance. We can cap the boost so that `score_final ≤ 2 × relevance` and apply the adjustment only when the baseline relevance is above the 60th percentile to avoid surfacing low-quality content purely for fairness. Finally, boosting is limited to creators whose language/region matches `ctx_locale`, keeping recommendations culturally relevant.

**Data used for training:** creator cumulative exposure counts (30-day), raw relevance signals, creator locale/language, user locale, historical click/watch feedback.

---

### 13. CO_STREAM_MATCH
**Request**
```json
{
  "creator_id": "c123"
}
```

**Response**
```json
{
  "partners": [
    { "creator_id": "c456", "synergy_score": 0.75 },
    { "creator_id": "c789", "synergy_score": 0.71 }
  ]
}
```

**Related Work**  
Grover, A., & Leskovec, J. (2016). *node2vec.* **KDD 2016**, 855-864. <https://arxiv.org/abs/1607.00653>

**Implementation Insight**  
node2vec embeddings of the creator-collab graph give cosine similarities; filter by overlapping schedules & category, average over five random-walk seeds to stabilise `synergy_score`.

**Data used for training:** creator collaboration/co-stream graph, viewer-overlap metrics, creator schedules, content categories, historical co-stream success metrics (viewer lift, retention).

---
