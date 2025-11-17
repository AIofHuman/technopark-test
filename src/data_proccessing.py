import pandas as pd
import os
import logging

logger = logging.getLogger('technopark-test-task')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

class DataProcessor:
    def __init__(
            self, 
            cat_features, target='target_unit_price_rub',
            id_feature = None,
            upper_target_bound = 350,
            columns_to_drop = [
                'labor_minutes_per_unit',
                'material_cost_rub',
                'labor_cost_rub',
                'unit_price_rub',
                'target_labor_min'
            ]
    ):
        self.target = target
        self.cat_features = cat_features
        self.id_feature = id_feature
        self.upper_target_bound = upper_target_bound
        self.num_features = None
        self.columns_to_drop = columns_to_drop

        
    def validate_and_clean(self, df, file_name_cleaned_data=None):
        """Валидация и очистка данных"""
        # Проверка наличия целевой переменной
        if self.target not in df.columns:
            raise ValueError(f"Target column {self.target} not found")
        
        df = df.drop(columns=self.columns_to_drop)
        
        self.num_features = list(set(df.columns.drop([self.id_feature, self.target])) - set(self.cat_features))

        logger.info(f"Identified {len(self.cat_features)} categorical features")
        logger.info(f"Identified {len(self.num_features)} numerical features")
            
        # Удаление полных дубликатов
        initial_shape = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_shape - len(df)} full duplicates")

        # Удаление дублей по id
        if self.id_feature != None:
            df = self._delete_duplicates_by_rfq_id(df)
        
        # Обработка пропущенных значений
        df = self._handle_missing_values(df)

        # отсекаем датасет по верхенму значению таргета
        if self.upper_target_bound != 0:
            before_cut = len(df)
            df = df[df[self.target]<=self.upper_target_bound]
            logger.info(f"Removed {before_cut - len(df)} records by upper target bound")
        
        if file_name_cleaned_data != None:
            full_path = os.path.join(PROJECT_DIR, "data", file_name_cleaned_data)
            df.to_csv(full_path,  index=False)
            logger.info(f"Save cleaned dataset to {full_path}, {len(df)} - records")

        return df.reset_index(drop=True)
    
    def _delete_duplicates_by_rfq_id(self, df):
        """Удаление дубликатов по id"""
        unique_count = df[self.id_feature].nunique()
        total_count = len(df)
        if total_count==unique_count:
            logger.info('All ids are unique in dataset')
            df = df.drop([self.id_feature], axis=1)
        else:
            duplicates_by_id = df[df.duplicated(subset=[self.id_feature], keep=False)]
            logger.info(f'Amount of id duplicates - {len(duplicates_by_id)}')
            df = df.groupby(self.id_feature).agg(
                lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
            ).reset_index(drop=True)
            logging.info(f'Removed - {total_count - len(df)} duplicates by {self.id_feature}')

        return df

    def _handle_missing_values(self, df):
        """Обработка пропущенных значений"""
        # Для числовых признаков - медиана
        for col in self.num_features:
            df[col] = df[col].fillna(df[col].median())
        
        # Для категориальных - мода
        for col in self.cat_features:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df
        
        