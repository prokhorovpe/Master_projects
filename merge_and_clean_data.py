import pandas as pd
import os

def merge_and_clean_datasets():
    """Объединяет старые и новые данные, оставляя только необходимые столбцы."""
    print("🚀 Начинаем объединение и очистку данных...")

    # Пути к файлам
    old_file_path = 'data/yandmast_obr2.csv'
    new_file_path = 'data/julyaug.csv'
    backup_path = 'data/yandmast_obr2_backup.csv'  # Создаем резервную копию на всякий случай

    # 1. Создаем резервную копию старого файла
    if os.path.exists(old_file_path):
        os.rename(old_file_path, backup_path)
        print(f"✅ Резервная копия создана: {backup_path}")

    # 2. Загружаем старые данные (только нужные столбцы)
    old_df = pd.read_csv(backup_path, usecols=['study_date', 'inventory_number', 'type_of_service'])
    print(f"✅ Загружено старых записей: {len(old_df):,}")

    # 3. Загружаем новые данные с правильной кодировкой и разделителем
    new_df = pd.read_csv(new_file_path, usecols=['study_date', 'inventory_number', 'type_of_service'], encoding='cp1251', sep=';')
    print(f"✅ Загружено новых записей ДО преобразования: {len(new_df):,}")
    
    # ✅✅✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Преобразование формата даты ✅✅✅
    # Пробуем преобразовать дату из формата DD.MM.YYYY в YYYY-MM-DD
    new_df['study_date'] = pd.to_datetime(new_df['study_date'], format='%d.%m.%Y', errors='coerce')
    # Преобразуем обратно в строку в нужном формате
    new_df['study_date'] = new_df['study_date'].dt.strftime('%Y-%m-%d')
    # Удаляем строки, где дата не была распознана
    new_df = new_df.dropna(subset=['study_date'])
    print(f"✅ Загружено новых записей ПОСЛЕ преобразования: {len(new_df):,}")
    # ✅✅✅ КОНЕЦ ИСПРАВЛЕНИЯ ✅✅✅

    # 4. Объединяем
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    print(f"✅ Всего записей после объединения: {len(combined_df):,}")

    # 5. Сохраняем обратно в старый файл (в UTF-8 для будущей совместимости)
    combined_df.to_csv(old_file_path, index=False, encoding='utf-8')
    print(f"✅ Данные успешно сохранены в: {old_file_path}")

    # 6. (Опционально) Удаляем резервную копию, если все прошло успешно
    # os.remove(backup_path)
    # print("🗑 Резервная копия удалена.")

if __name__ == "__main__":
    merge_and_clean_datasets()
    print("🎉 Готово! Теперь можно запускать run_pipeline.py")