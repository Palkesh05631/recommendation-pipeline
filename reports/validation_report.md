# Data Validation Report

| Table | Total rows | Valid | Rejected | % rejected |
|---|---|---|---|---|
| users | 948 | 943 | 5 | 0.53% |
| items | 1687 | 1682 | 5 | 0.30% |
| interactions | 100109 | 99978 | 131 | 0.13% |

## users: violations per check

| Check | Rows violating |
|---|---|
| missing_user_id | 0 |
| invalid_dtype_user_id | 0 |
| invalid_dtype_age | 0 |
| invalid_age | 2 |
| duplicate_user_id | 3 |

## items: violations per check

| Check | Rows violating |
|---|---|
| missing_item_id | 0 |
| invalid_dtype_item_id | 0 |
| missing_title | 2 |
| duplicate_item_id | 3 |
| invalid_genre_flag | 0 |

## interactions: violations per check

| Check | Rows violating |
|---|---|
| missing_user_id | 0 |
| missing_item_id | 0 |
| missing_rating | 10 |
| missing_timestamp | 0 |
| invalid_dtype_user_id | 0 |
| invalid_dtype_item_id | 0 |
| invalid_dtype_rating | 8 |
| invalid_dtype_timestamp | 0 |
| invalid_rating | 25 |
| invalid_timestamp | 16 |
| orphan_user_id | 10 |
| orphan_item_id | 10 |
| duplicate_interaction | 52 |

_A row can violate several checks, so per-check counts may sum to more than the number of rejected rows. Rejected rows are stored in `data/errors/` with an `error_reason` column._