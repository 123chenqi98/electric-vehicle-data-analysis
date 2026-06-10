import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）\s]', '', text)
    text = ' '.join(text.split())
    return text.strip()

def main():
    labels_path = r'C:\Users\cqzuishuai\Desktop\project\data_analysis\data\raw\Online-Reviews-with-Sentiment-Labels.csv'
    no_labels_path = r'C:\Users\cqzuishuai\Desktop\project\data_analysis\data\raw\Online-Reviews-without-Sentiment-Labels.csv'
    
    try:
        df_labels = pd.read_csv(labels_path, encoding='gbk')
    except:
        df_labels = pd.read_csv(labels_path, encoding='utf-8', errors='ignore')
    
    try:
        df_no_labels = pd.read_csv(no_labels_path, encoding='gbk')
    except:
        df_no_labels = pd.read_csv(no_labels_path, encoding='utf-8', errors='ignore')
    
    df_labels['评论'] = df_labels.iloc[:, 2].apply(clean_text)
    df_labels['情感标签'] = df_labels.iloc[:, 3].fillna(-1).astype(int)
    df_labels['车型ID'] = df_labels.iloc[:, 1]
    
    df_no_labels['评论'] = df_no_labels.iloc[:, 1].apply(clean_text)
    
    df_labels.to_csv('data/processed/reviews_with_labels.csv', index=False, encoding='utf-8-sig')
    df_no_labels.to_csv('data/processed/reviews_without_labels.csv', index=False, encoding='utf-8-sig')
    print('数据修复完成')

if __name__ == '__main__':
    main()
