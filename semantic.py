# ======================================== L3T12 Compulsory Task 1 ========================================
import spacy
nlp = spacy.load('en_core_web_md')

# Write a note about what you found interesting about the similarities between cat, monkey and banana:
'''Taking a look at the cosine similairty scores, I noticed that when an item is compared to itself it gets a score of 1,
which confirms that they are most similar (identical). I then noticed that cat and monkey have a score of 0.54 which is closer to 1 than 0.
So they are similar probably because they are both animals. Then I saw that apple and banana are similar with a score 0.58 likely because they
are both fruit. What is interesting is that spacy registered a higher similarity between monkey and banana, with a score of 0.45 than the other
animal-fruit combinations. This is likely because we associate monkeys with eating banana's.'''

# An examples of my own comparing umbrella, rain and river:
word4 = nlp("umbrella")
word5 = nlp("rain")
word6 = nlp("river")
print(word4.similarity(word5))
print(word6.similarity(word5))
print(word6.similarity(word4))
'''What I notice here is that river and rain were the most similar with a score of 0.4. Umbrella and rain had a score of
0.18 which was slightly higher than umbrella and river (0.11). This is likely because an umbrella is useful in the rain, but 
and umbrella and a river don't have much to do with one another.'''

# Run the example file with the simpler language model ‘en_core_web_sm’:
'''I ran the example file in the simpler language model and the similarity scores were so different! The monkey and banana
had a score of 0.4 indicating some similarity, but then combinations that didn't show similarity before like apple and cat, then
had high similarity of 0.7! In real life, a cat and and apple don't have a stronger association than a monkey and banana.
This shows that the simpler language model is less accurate.'''