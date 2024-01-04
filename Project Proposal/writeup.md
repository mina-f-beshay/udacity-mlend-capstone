## Domain Background

## Problem Statement

## Solution Statement

## Datasets and Inputs

The dataset consists of 3 main files
1. profile.json: Rewards program users (17000 users x 5 fields)
    | Column | Dataype | Description
    | -------|---------|-----------
    | gender | categorical | M, F, O, or null
    | age    | numeric | missing value encoded as 118
    | id     | string/hash | 
    | became_member_on | date |  format YYYYMMDD
    | income | numeric
2. portfolio.json: Offers sent during 30-day test period (10 offers x 6 fields)
    | Column | Datatype |Description
    | -------|----------|------------
    | reward | numeric  | money awarded for the amount spent
    | channels| list    | web, email, mobile, social
    | difficulty| numeric | money required to be spent to receive reward
    | duration| numeric | time for offer to be open, in days
    | offer_type| string | bogo, discount, informational
    | id| string/hash |
3. transcript.json: Event log (306648 events x 4 fields)
    | Column | Datatype |Description
    | -------|----------|------------
    | person | string/hash |
    | event  | string | offer received, offer viewed, transaction, offer completed
    | value  | dictionary | different values depending on event type
    | offer id| string/hash | not associated with any "transaction"
    | amount| numeric | money spent in "transaction"
    | reward | numeric | money gained from "offer completed"
    | time| numeric | hours after start of test
## Benchmark Model

## Evaluation Metrics

## Presentaion

## Project Design