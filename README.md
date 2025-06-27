# Stream Recommendation System

A production-ready proof-of-concept for intelligent live stream recommendations, starting with the **HOMETOWN** scenario that prioritizes streams from the user's geographic region.

## 🚀 Quick Start & Evaluation

```bash
# Setup and start API
uv sync && uv run python run_server.py

# Run comprehensive evaluation suite
uv run python evaluate_recommendations.py    # nDCG@5, Precision@5, Recall@5
uv run python detailed_comparison.py         # Algorithm behavior analysis  
uv run python test_both_endpoints.py         # Side-by-side comparison
```

**Evaluation Results**: Both basic and ML-enhanced algorithms achieve 0.196 nDCG@5 with <3ms response times on all 198 test users. See [`COMPREHENSIVE_ACCURACY_REPORT.md`](COMPREHENSIVE_ACCURACY_REPORT.md) for detailed metrics and [`EVALUATION_SUMMARY.md`](EVALUATION_SUMMARY.md) for complete results.

---

## Task: [Streaming Service Scenarios](material/Task.md)

After analysing the above linked task, in addition to the initially suggested 2 **Scenarios**, 11 additional ones were identified and could be considered for implementation as they positevly impact business KPIs (e.g., watch time, session length, ad revenue, etc.).


### Overview of possible Scenarios

| #  | Scenario              | Detailed Description                                                                                                                        | Key KPI(s) Tracked & Definition                                                                                              |
|----|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| 1  | **HOMETOWN**          | Order the front-page grid so that live channels **from the user’s own city/region** appear first.                                           | **CTR** – on location-matched stream impressions                                          |
| 2  | **SMS_ALERT**         | Nightly job that texts users who have been **inactive > 30 days** with a personalised “Your favourite streamer is live” reminder.           | **Reactivation** – number of dormant users who open the link and count as daily active users the next day               |
| 3  | **HOME_FEED**         | Personalise the homepage carousel using viewing history, follows, and embeddings.                                                           | **CTR** (homepage clicks / impressions); Total watch-hours per session                                                       |
| 4  | **CATEGORY_FEED**     | Re-rank all live streams inside a selected game/genre page for each user.                                                                   | Streams-per-visit (depth); Average session length                                                                            |
| 5  | **UP_NEXT**           | Autoplay the best next live channel when the viewer leaves or a stream ends.                                                                | Average additional watch-minutes;                                                      |
| 6  | **SEARCH_SUGGEST**    | Re-order search results and power type-ahead suggestions with semantic & behavioural relevance.                                             | Search CTR (result clicks / impressions): Search success rate (queries that lead to a play)                              |
| 7  | **FOLLOW_RECOMMEND**  | Suggest creators to follow based on graph similarity and social proof.                                                                      | Follows/user/day; 7-day retention                                                                                |
| 8  | **CLIP_DIGEST**       | Daily email/push digest bundling personalised clips the user missed while offline.                                                          | Digest open rate; Weekly returning users; Incremental watch-hours                                                      |
| 9  | **RAID_SUGGEST**      | At stream-end, recommend target channels for creators to **raid** based on audience overlap and diversity criteria.                         | Post-raid viewer retention (%); Exposure share of small creators                                                         |
| 10 | **LIVE_NOTIFY**       | Real-time push/SMS when a followed creator goes live, throttled by quiet hours.                                                             | Notification open rate; Sessions initiated via notification                                                              |
| 11 | **DISCOVERY_ROW**     | Inject a “Because you watched X” horizontal row of related live channels beneath the player.                                                | • Additional streams watched/visit  • Incremental watch-minutes                                                              |
| 12 | **COLD_CREATOR_BOOST**| Dedicated carousel that boosts new or low-exposure creators while maintaining relevance.                                                    | Exposure Gini (inequality of impressions); Click-through rate on boosted carousel                                        |
| 13 | **CO_STREAM_MATCH**   | Recommend synergistic creators for **co-streaming** (Squad Streams) based on audience overlap and schedule match.                           | Number of co-streams formed; Combined watch-hours generated by co-streams                                               |





---

For a more detailed technical specifications with API parameters and possible implementation based on published related work for each scenario check out: 
### [--> Scenario Details <--](material/Scenarios.md)

A more detailed description of an implemented POC for the HOMETOWN recommendation scenario can be found:
### [--> HOMETOWN POC Implementation <--](material/POC.md)



