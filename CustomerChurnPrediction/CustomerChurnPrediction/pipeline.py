from src.data import load_data
from src.preprocess import main_pipeline, test_pipeline
from sklearn.pipeline import Pipeline
from src.feature_engineering import feature_engineering_pipeline
import pandas as pd
import os

def create_pipeline():
    pipeline = Pipeline([
        ('preprocessor', main_pipeline())
    ])
    
    return pipeline

if __name__ == "__main__":
    pipeline = create_pipeline()
    fe_pipeline = feature_engineering_pipeline()
    test_pipe = test_pipeline()

    data = load_data(os.path.join('data', 'raw', 'train.csv'))
    test_data = load_data(os.path.join('data', 'raw', 'test.csv'))
    processed_data = pipeline.fit_transform(data.drop('id', axis=1))
    processed_test_data = test_pipe.fit_transform(test_data.drop('id', axis=1))

    processed_data_df = pd.DataFrame(processed_data, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
    processed_test_data_df = pd.DataFrame(processed_test_data, columns=test_pipe.named_steps['preprocessor'].get_feature_names_out())

    processed_test_data_df.to_csv(os.path.join('data', 'processed', 'processed_test.csv'), index=False)
    processed_data_df.to_csv(os.path.join('data', 'processed', 'processed_train.csv'), index=False)

    # feature_selected_data = fe_pipeline.fit_transform(processed_data_df.drop('cat__ordinal__Churn', axis=1), processed_data_df['cat__ordinal__Churn'])
    # feature_selected_data_df = pd.DataFrame(feature_selected_data, columns=fe_pipeline.named_steps['feature_selector'].get_feature_names_out())

    # feature_selected_test_data = processed_test_data_df[fe_pipeline.named_steps['feature_selector'].get_feature_names_out()]
    # feature_selected_test_data.to_csv(os.path.join('data', 'processed', 'feature_selected_test.csv'), index=False)

    # feature_selected_data_df = processed_data_df[fe_pipeline.named_steps['feature_selector'].get_feature_names_out()]
    # feature_selected_data_df['Churn'] = processed_data_df['cat__ordinal__Churn'].values
    # feature_selected_data_df.to_csv(os.path.join('data', 'processed', 'feature_selected_train.csv'), index=False)

    # print("Selected Features DataFrame:")
    # print(feature_selected_data_df.head())