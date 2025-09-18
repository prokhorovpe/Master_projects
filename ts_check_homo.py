import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from config import PROCESSED_DATA_PATH, TEST_START_DATE

def analyze_service_time_series():
    """Анализирует обработанные временные ряды на предмет наличия данных в тестовом периоде и однородности."""
    print("🔍 Анализ service_time_series.pkl...")
    
    # Загружаем обработанные данные
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        service_time_series = pickle.load(f)
    
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = test_start + pd.offsets.MonthEnd(0)  # Последний день августа 2025
    train_start = pd.to_datetime('2024-01-01')  # Начало тренировочного периода
    train_end = test_start - pd.Timedelta(days=1)  # Конец тренировочного периода

    print(f"Тренировочный период: с {train_start.date()} по {train_end.date()}")
    print(f"Тестовый период: с {test_start.date()} по {test_end.date()}")
    print("=" * 80)

    for modality, ts in service_time_series.items():
        print(f"\n📊 Модальность: {modality}")
        
        # Универсальная обработка данных
        if isinstance(ts, pd.Series):
            df = ts.reset_index()
            df.columns = ['ds', 'y']
        elif isinstance(ts, pd.DataFrame):
            if len(ts.columns) >= 2:
                df = ts.iloc[:, :2].copy()
                df.columns = ['ds', 'y']
            else:
                df = ts.reset_index()
                df.columns = ['ds', 'y']
        else:
            print(f"❌ Неподдерживаемый тип данных: {type(ts)}")
            continue

        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce').clip(lower=0)
        df = df.dropna(subset=['y']).reset_index(drop=True)

        # Разделяем данные на периоды
        train_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)].copy()
        test_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].copy()

        # Базовые статистики
        print(f"   Всего дней в тестовом периоде: {len(test_df)}")
        print(f"   Сумма исследований в тестовом периоде: {test_df['y'].sum():,}")
        print(f"   Минимальное значение: {test_df['y'].min():,}")
        print(f"   Максимальное значение: {test_df['y'].max():,}")
        print(f"   Количество дней с нулевыми значениями: {(test_df['y'] == 0).sum()}")

        # Анализ однородности
        if len(train_df) > 0 and len(test_df) > 0:
            print("\n   📈 АНАЛИЗ ОДНОРОДНОСТИ ДАННЫХ:")
            print("   " + "-" * 50)
            
            # 1. Сравнение средних значений
            train_mean = train_df['y'].mean()
            test_mean = test_df['y'].mean()
            mean_diff_pct = ((test_mean - train_mean) / train_mean) * 100
            print(f"   Среднее значение:")
            print(f"   - Тренировочный период: {train_mean:.2f}")
            print(f"   - Тестовый период: {test_mean:.2f}")
            print(f"   - Разница: {mean_diff_pct:+.2f}%")
            
            # 2. Сравнение стандартных отклонений
            train_std = train_df['y'].std()
            test_std = test_df['y'].std()
            std_diff_pct = ((test_std - train_std) / train_std) * 100
            print(f"   Стандартное отклонение:")
            print(f"   - Тренировочный период: {train_std:.2f}")
            print(f"   - Тестовый период: {test_std:.2f}")
            print(f"   - Разница: {std_diff_pct:+.2f}%")
            
            # 3. Статистический тест (t-test для средних)
            if len(train_df) >= 30 and len(test_df) >= 30:  # Минимальный размер для t-test
                t_stat, p_value = stats.ttest_ind(train_df['y'], test_df['y'])
                print(f"   T-test (p-value): {p_value:.4f}")
                if p_value < 0.05:
                    print("   ⚠️  Статистически значимое различие средних!")
                else:
                    print("   ✅ Нет статистически значимого различия средних")
            
            # 4. Сравнение медиан
            train_median = train_df['y'].median()
            test_median = test_df['y'].median()
            median_diff_pct = ((test_median - train_median) / train_median) * 100
            print(f"   Медиана:")
            print(f"   - Тренировочный период: {train_median:.2f}")
            print(f"   - Тестовый период: {test_median:.2f}")
            print(f"   - Разница: {median_diff_pct:+.2f}%")
            
            # 5. Коэффициент вариации
            train_cv = (train_std / train_mean) * 100 if train_mean > 0 else 0
            test_cv = (test_std / test_mean) * 100 if test_mean > 0 else 0
            print(f"   Коэффициент вариации:")
            print(f"   - Тренировочный период: {train_cv:.2f}%")
            print(f"   - Тестовый период: {test_cv:.2f}%")
            
            # 6. Сезонность (сравнение по дням недели)
            if len(train_df) > 7 and len(test_df) > 7:
                train_df['day_of_week'] = train_df['ds'].dt.dayofweek
                test_df['day_of_week'] = test_df['ds'].dt.dayofweek
                
                train_dow_means = train_df.groupby('day_of_week')['y'].mean()
                test_dow_means = test_df.groupby('day_of_week')['y'].mean()
                
                dow_correlation = train_dow_means.corr(test_dow_means)
                print(f"   Корреляция паттернов по дням недели: {dow_correlation:.3f}")

        # Визуализация
        try:
            plt.figure(figsize=(12, 8))
            
            # Полный временной ряд
            plt.subplot(2, 1, 1)
            plt.plot(df['ds'], df['y'], label='Полный ряд', alpha=0.7)
            plt.axvline(x=train_end, color='red', linestyle='--', label='Начало тестового периода')
            plt.title(f'Временной ряд: {modality}')
            plt.xlabel('Дата')
            plt.ylabel('Количество исследований')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Сравнение распределений
            plt.subplot(2, 1, 2)
            plt.hist(train_df['y'], bins=30, alpha=0.7, label='Тренировочный период', density=True)
            plt.hist(test_df['y'], bins=30, alpha=0.7, label='Тестовый период', density=True)
            plt.title('Сравнение распределений')
            plt.xlabel('Количество исследований')
            plt.ylabel('Плотность')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{modality}_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 График сохранен как: {modality}_analysis.png")
            
        except Exception as e:
            print(f"   ⚠️  Ошибка при построении графиков: {e}")

    print("\n✅ Анализ завершен.")

if __name__ == "__main__":
    analyze_service_time_series()