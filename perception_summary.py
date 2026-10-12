# perception_summary.py
import pandas as pd


def summarize_perceptions():
  print('Generating Market Perception Summary across Pillars...')
  try:
    df = pd.read_csv('master_nlp_aspects.csv')
  except FileNotFoundError:
    print(
        "[Error] 'master_nlp_aspects.csv' not found. Run aspect sentiment"
        ' first.'
    )
    return

  pillars = ['Cost & ROI', 'Security & Privacy', 'Scalability & Tech', 'Sustainability & Green']
  
  summary = df.groupby('sector')[pillars].mean()
  print('\n--- Average VADER Sentiment per Sector & Pillar ---')
  print(summary)

  print('\nPerception Summary generated successfully.')


if __name__ == '__main__':
  summarize_perceptions()