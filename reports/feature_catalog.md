# Feature Catalog

| Entity | Feature | Type | Description | Source | Logic |
|---|---|---|---|---|---|
| user | user_avg_rating | float | Mean rating the user gives | interactions.rating | mean(rating) per user |
| user | user_activity_count | int | Number of interactions by the user | interactions | count(*) per user |
| user | user_rating_std | float | Spread of the user's ratings (0 if one rating) | interactions.rating | std(rating) per user |
| user | user_pct_high_ratings | float | Share of ratings >= 4 | interactions.rating | mean(rating>=4) per user |
| user | preferred_genre | category | Genre the user interacts with most | interactions + items genre flags | argmax of summed genre flags |
| user | days_since_last_interaction | float | Recency of the user's last activity | interactions.datetime | reference_date - max(datetime) |
| item | item_popularity | int | Number of ratings the item received | interactions | count(*) per item |
| item | item_avg_rating | float | Mean rating the item received (global mean if unseen) | interactions.rating | mean(rating) per item |
| item | item_popularity_pct | float | Popularity percentile (0-1) | item_popularity | rank(pct=True) |
| item | item_age_years | float | Age of the item at reference date | items.release_year | reference_year - release_year |
| item | n_genres | int | Number of genres tagged on the item | items genre flags | sum(genre flags) |
| item | <genre flags x19> | int(0/1) | One-hot genre vector used for diversity / content signals | items | as provided |
| interaction | rating_minus_user_avg | float | Rating relative to the user's own average | interactions + user_avg_rating | rating - user_avg_rating |
| interaction | rating_minus_item_avg | float | Rating relative to the item's average | interactions + item_avg_rating | rating - item_avg_rating |
| interaction | days_since_interaction | float | Age of the interaction at reference date | interactions.datetime | reference_date - datetime |
| interaction | is_high_rating | int(0/1) | Binary relevance label | interactions.rating | rating >= 4 |

All features are computed from the **training split only** at a fixed `reference_date` (the latest training timestamp), so training and prediction see identical definitions.