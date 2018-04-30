from pythonrouge.pythonrouge import Pythonrouge

# system summary(predict) & reference summary
summary = [[" Tokyo is the one of the biggest city in the world."]]
reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

# initialize setting of ROUGE to eval ROUGE-1, 2, SU4
# if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
# if recall_only=True, you can get recall scores of ROUGE
rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)
