import pandas as pd
from preprocess import review_segmentation_w_date, seperate_review_by_sentiment
from model import get_textrank_mod, baseline_model


def preprocess_reviews(review_df: pd.DataFrame, n=5):
    '''
    Transform raw review data into summarize-ready data

    parameters
        review_df: pd.DataFrame

        Input DataFrame Columns:
            id: review ID
            reviewed_item_id: reviewed item id
            summary: summary of the review
            description: review description
            reviewed_date: reviewed date

        n: Number of wanted summary review sentences

    returns
        pd.DataFrame

        Output DataFrame Columns:
            restaurant_id: restaurant id with checker
            reviews: list of positive reviews.
            ids: list of review id.
            personalize: dict used to personalize the pagerank algorithm (weight sentence that have food word).
    '''

    review_df.loc[:, 'review'] = review_df.description
    review_df.loc[(review_df.summary.str.len() > review_df.description.str.len()) | (
        review_df.description.isna()), 'review'] = review_df.summary
    review_df = review_df.drop(
        review_df[review_df.review.str.len() < 15].index)
    review_df = review_df.groupby(
        by='reviewed_item_id').filter(lambda x: len(x) >= 15)

    processed_reviews = []
    for i, text in review_df.groupby(by='reviewed_item_id'):
        review_list = text['review']
        review_id = text['id'].values
        reviewed_item_id = str(i)
        reviewed_date = text['reviewed_date'].values
        all_sentences, all_ids, all_review_date = review_segmentation_w_date(
            review_list, review_id, reviewed_date)
        pos_reviews, pos_ids, personalize = seperate_review_by_sentiment(
            all_sentences, all_ids, all_review_date, is_personalize=True)
        if len(pos_reviews) >= n*3:
            processed_reviews.append(
                {'reviewed_item_id': reviewed_item_id, 'reviews': pos_reviews, 'ids': pos_ids, 'personalize': personalize})
    return pd.DataFrame(processed_reviews)


def summarize_reviews(processed_review_df, n=5, is_dup_id=True, patient=3, drop_redundant=True, threshold=0.7):
    '''
    Generate review summary for each reviewed_item_id

    parameters
        processed_review_df: pd.DataFrame with columns:
            reviewed_item_id: reviewed item id
            reviews: list of positive reviews.
            ids: list of review id.
            personalize: dict used to personalize the pagerank algorithm (weight sentence that have food word).
        n: number of summarize reviews (default:10)
        is_dup_id: if you want duplicate id or not if True the review return could have the same review ID
        patient: number of max iteration when skipping redundat sentences and duplicate review ID
        drop_redundant: if True the sentences that have simliarity score higher than threshold, compare to selected review, will be drop
        threshold: used to drop the redundant sentences

    returns
        pd.DataFrame
            Summarized reviews with columns
            - id: reviewed item id
            - review_summary: list of summary dict
    '''
    output = []
    for _, row in processed_review_df.iterrows():
        try:
            review = get_textrank_mod(row['reviews'], row['ids'], n=n, is_dup_id=is_dup_id, patient=patient,
                                      drop_redundant=drop_redundant, threshold=threshold, personalize=row['personalize'])
        except Exception as e:
            review = baseline_model(row['reviews'], row['ids'], n=n, is_dup_id=is_dup_id,
                                    patient=patient, drop_redundant=drop_redundant, threshold=threshold)
        output.append({'id': row['reviewed_item_id'],
                      'review_summary': review})
    return pd.DataFrame(output)
