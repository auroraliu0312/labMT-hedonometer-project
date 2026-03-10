# labMT Hedonometer Dataset Analysis

# Project Overview

This project analyzes the labMT 1.0 dataset, which contains happiness scores for 10,222 English words rated by Amazon Mechanical Turk workers. The dataset enables measurement of emotional valence in large-scale texts across four different corpora: Twitter, Google Books, NY Times, and song lyrics. Our analysis combines quantitative exploration (distributions, disagreements, corpus overlaps) with qualitative interpretation of selected words to understand how emotional meaning varies across contexts and communities.

## Dataset

- Source

The dataset comes from Dodds et al. (2011) "Temporal Patterns of Happiness and Information in a Global Social Network:  Hedonometrics and Twitter," published in PLOS ONE. It was constructed by collecting frequency rankings from four corpora and crowdsourcing happiness ratings for each word via Amazon Mechanical Turk.

- Data Dictionary

We created a data dictionary to summarize each column's content, type, and missing values.

| Column                      | Type    | Missing Values | Description                              |
|----------------------------| ---------|----------------|------------------------------------------|
| word                        | str     | 0              | Word being assessed                      |
| happiness_rank              | int64   | 0              | Rank based on happiness (1 = happiest)   |
| happiness_average           | float64 | 0              | Average happiness score (1-9)            |
| happiness_standard_deviation| float64 | 0              | Standard deviation of happiness          |
| twitter_rank                | float64 | 5222           | Twitter rank of the word                 |
| google_rank                 | float64 | 5222           | Google Books rank of the word            |
| nyt_rank                    | float64 | 5222           | New York Times rank of the word          |
| lyrics_rank                 | float64 | 5222           | Lyrics rank of the word                  |

> Missing ranks (`NaN`) indicate that the word does not appear in that corpus's top 5,000 most frequent words.

# Methods

We performed the following analyses using Python with pandas, matplotlib, and numpy:

## 1.1 Load the File

We loaded the labMT 1.0 dataset using pandas `read_csv`. The dataset is tab-delimited and contains three lines of metadata at the top, which we skipped using `skiprows=3`. We also treated '--' as missing values (`NaN`) using `na_values="--"`.

The dataset contains 10222 rows and 8 columns.  

A missing rank (`--`) indicates that the word does not appear in that particular corpus.

## 1.2 Data Dictionary (See "DataSet": Data Dictionary)

## 1.3 Sanity Checks

 We performed several sanity checks to ensure the dataset is clean and reasonable.

## Check for duplicated words:
 There are no duplicated words in the dataset, confirming unique entries for each word.

## Random sample of 15 rows:
 We inspected a random subset of 15 rows to verify that values appear consistent and correct. Example sample:

| word | happiness_rank | happiness_average | happiness_standard_deviation | twitter_rank | google_rank | nyt_rank | lyrics_rank |
|------|----------------|-------------------|------------------------------|--------------|-------------|----------|--------------|
| prom | 2883 | 5.94 | 1.3763 | 4876.0 | NaN | NaN | NaN |
| on | 4515 | 5.56 | 1.0721 | 13.0 | 16.0 | 10.0 | 14.0 |
| mis | 7718 | 4.88 | 1.0999 | 4517.0 | NaN | NaN | 1292.0 |
| friendship | 34 | 7.96 | 1.1241 | 4273.0 | 3098.0 | 3669.0 | 3980.0 |
| naval | 4925 | 5.48 | 1.2493 | NaN | 3295.0 | 4436.0 | NaN |
| grand | 533 | 7.06 | 1.3614 | 1685.0 | 1709.0 | 944.0 | 1575.0 |
| wen | 8029 | 4.80 | 1.0498 | 1345.0 | NaN | NaN | NaN |
| extract | 5861 | 5.28 | 1.4574 | NaN | 4832.0 | NaN | NaN |
| harry | 6055 | 5.24 | 1.2545 | 2313.0 | 3856.0 | 1692.0 | NaN |
| designers | 1544 | 6.38 | 1.4831 | NaN | NaN | 3890.0 | NaN |
| external | 4895 | 5.48 | 1.2162 | NaN | 1259.0 | NaN | NaN |
| screwed | 9685 | 3.24 | 1.6107 | 4145.0 | NaN | NaN | 4908.0 |
| pittsburgh | 6533 | 5.14 | 1.3852 | NaN | NaN | 2038.0 | NaN |
| vital | 3609 | 5.76 | 1.5592 | NaN | 2732.0 | 2165.0 | NaN |
| obedience | 5327 | 5.40 | 1.6162 | NaN | 4840.0 | NaN | NaN |


## Top 10 positive words:
The words with the highest happiness scores are logical and correspond to highly positive terms. 

| word      |happiness_average|
|-----------|-----------------|
| laughter  | 8.50            |
| happiness | 8.44            |
| love      | 8.42            |
| happy     | 8.30            |
| laughed   | 8.26            |
| laugh     | 8.22            |
| laughing  | 8.20            |
| excellent | 8.18            |
| laughs    | 8.18            |
| joy       | 8.16            |

## Top 10 negative words:
The words with the lowest happiness scores correspond to negative or sensitive terms.

| word       | happiness_average |
|-----------|-----------------|
| suicide   | 1.30            |
| terrorist | 1.30            |
| rape      | 1.44            |
| murder    | 1.48            |
| terrorism | 1.48            |
| cancer    | 1.54            |
| death     | 1.54            |
| died      | 1.56            |
| kill      | 1.56            |
| killed    | 1.56            |

> These checks confirm that the happiness scores and words are reasonable, and no data entry errors are apparent.

# Results

## 2.1 Distribution of Happiness Scores

![Figure 1: Distribution of Happiness Scores](figures/happiness_average_hist.png)

Summary Statistics:
- Mean: 5.38
- Median: 5.44
- Standard Deviation: 1.08
- 5th Percentile: 3.18
- 95th Percentile: 7.08

The distribution of happiness scores is centered slightly above 5, with mean and median very close (5.38 and 5.44), indicating approximate symmetry. Most words fall between 4.5 and 6.5, suggesting that everyday English vocabulary leans mildly positive. Extremely positive and extremely negative words are relatively rare, with only 5% of words scoring below 3.18 and 5% scoring above 7.08. This pattern suggests that common language tends toward moderate positivity, with strong emotional words occupying the tails of the distribution.

 One interesting pattern is the distribution reveals the negative tail (scores below 3.18) is slightly longer than the positive tail (scores above 7.08). This means that when words do deviate from the neutral range, they are slightly more likely to be negative than positive. However, the overall mass of the distribution sits in the 5-6 range, indicating that everyday language maintains a mild positivity bias. This suggests that English vocabulary has a wider range of mildly negative words, but the most intensely positive words are more extremely positive than the most intensely negative words are extremely negative. 
 
 ![Figure 1b: Distribution with Highlighted Tails and Extremes](figures/happiness_distribution_enhanced.png)
 
 According to the advanced figure 1 below, a closer examination of the tails reveals an interesting asymmetry. The negative tail extends from 1 to 3.18, spanning 2.18 points, while the positive tail extends from 7.08 to 9, spanning only 1.92 points. This means that when words deviate from the neutral range, they are slightly more likely to be negative than positive English vocabulary. However, the extremes tell a different story. The most positive word "laughter" (8.50) lies 3.12 points above the mean, while the most negative word "suicide" (1.30) lies 4.08 points below the mean. This indicates that although there are more mildly negative words, the most intensely negative words reach further from neutrality than the most intensely positive words. Overall, these patterns suggest that English vocabulary is structured with a broad spectrum of mild negativity but reserves its most extreme emotional intensity for positive expression. The strongly negative words (very low scores) are much less common than neutral or slightly positive words. This suggests that common language tends to lean slightly positive overall.

## 2.2 Disagreement: Words with High Standard Deviation

We used happiness_standard_deviation to measure how much people disagreed when rating each word.
![Figure 2: Happiness Average vs Standard Deviation](figures/happiness_vs_std_scatter.png)

We plotted a scatterplot with:
happiness_average on the x-axis
happiness_standard_deviation on the y-axis

Most words cluster in the middle of the plot. Their average happiness lies between roughly 4 and 7, and their standard deviation is around 1.0. This indicates that for the majority of words, annotators broadly agree on whether the word feels positive, neutral, or negative. In contrast, a small group of words have very high standard deviations (above 2.4). These “contested” words are those where annotators’ ratings strongly disagree.

Five examples include:
1. fucking / fuck / fuckin / fucked
These are very frequent swear words in contemporary English. They can signal strong negative emotion (“fucking awful”), but also serve as intensifiers in positive or humorous contexts (“that was fucking amazing”). Some annotators may rate them as very negative because of their taboo/insulting usage, while others may focus on their role as casual emphasis and assign more neutral or even mildly positive ratings. This mixture of offensiveness and playful emphasis likely produces the very high standard deviations we see.

2. whiskey (5.72, 2.64)
On the surface, “whiskey” is a relatively neutral object word. However, it is associated both with positive contexts (celebration, relaxation, craft culture) and negative ones (addiction, hangovers, self-destructive behavior). People who associate it with convivial, social drinking might rate it as positive, while others who associate it with alcoholism or “drinking to cope” might rate it as negative. This ambivalence around alcohol fits its high standard deviation.

3. churches (5.70, 2.46)
“Churches” has an average happiness slightly above 5, but a very large standard deviation. For some annotators, churches may evoke community, comfort, and spirituality; for others, they may evoke hypocrisy, exclusion, or painful personal experiences. Because religion is a deeply personal and culturally contingent topic, it makes sense that the emotional charge of “churches” varies widely across raters.

4. capitalism (5.16, 2.45)
“Capitalism” sits near the middle in average happiness, but with large disagreement. This reflects contemporary political and ideological divisions. Some annotators may view capitalism as synonymous with opportunity, innovation, and freedom. However, others may associate it with inequality, exploitation, and crisis. The word is strongly politicized, so we should expect its emotional valence to differ substantially across individuals.

5. pussy (4.80, 2.67)
This word is highly polysemous and gendered. It can be used as an insult (especially towards men, implying weakness), as a sexual term, and in some contexts as a reclaimed or playful expression. Different annotators may respond to different senses and social norms around sexism and sexuality, leading to wide disagreement in how “happy” or “unhappy” the word feels.

Overall, these words may be contested because:

- They can have multiple meanings (ambiguity)
- Their emotional tone depends on context
- Some may function as slang
- Some may carry irony or mixed connotations

The quantitative pattern (high standard deviation) reflects qualitative ambiguity. Words that allow multiple interpretations naturally produce more disagreement among raters. In this sense, standard deviation does not merely capture rating noise, it indexes cultural contestation and semantic instability.

## 2.3 Corpus comparison: rank coverage and overlaps

We created a heatmap to present the overlaps between corpora
![Figure 3: Corpus Overlap Heatmap](figures/corpus_overlap_heatmap.png)
This is a heatmap-like overlap matrix. Diagonal cells are 5000 by construction (top-5000 size). Off-diagonal cells show how many words appear in both corpora’s top-5000 lists.

The corpora share a substantial “core vocabulary,” but overlaps vary a lot depending on the pair:
•	NYT ∩ Google Books is relatively high (3414) → both are more formal/edited writing, so their frequent vocabulary overlaps more.
•	NYT ∩ Lyrics is relatively low (2241) → lyrics include more colloquial, stylized, and genre-specific vocabulary that doesn’t appear as often in newspaper prose.
•	Twitter overlaps strongly with Lyrics (3127) → both contexts are more conversational and informal, so they share more common slang / everyday terms.

We also created a Scatterplot of Twitter rank vs NYT rank for words that appear in both corpora. Lower rank means more frequent.
![Figure 4: Twitter vs NYT Rank](figures/twitter_vs_nyt_rank_scatter.png)

If the same words were similarly “common” across corpora, points would cluster near a diagonal trend. Instead, the plot shows wide spread: many words are very common in Twitter but not in NYT (and vice versa). That suggests “common language” is not an absolute property of a word—it is contextual, shaped by genre, platform norms, and institutional style (e.g., conversational talk vs editorial writing).

Concrete example of corpus-specific difference: “capitalism.”
It appears in Twitter and NYT but is much less prominent in Lyrics. This reflects communicative differences:
	•	Twitter and NYT contain political and institutional discourse.
	•	Lyrics foreground personal emotion, identity, and narrative voice rather than institutional vocabulary.
Similarly, slang or profanity terms (e.g., “fucking”) tend to appear in Twitter and Lyrics but are less common in formal corpora like Google Books, reflecting editorial filtering and stylistic norms.

# Qualitative “exhibit” of words

## 3.1 Build a small “exhibit” of words

| category | word | happiness_average | happiness_standard_deviation | twitter_rank | google_rank | nyt_rank | lyrics_rank |
|----------|------|-------------------|------------------------------|--------------|-------------|----------|--------------|
| very_positive | laughter | 8.50 | 0.9313 | 3600.0 | NaN | NaN | 1728.0 |
| very_positive | happiness | 8.44 | 0.9723 | 1853.0 | 2458.0 | NaN | 1230.0 |
| very_positive | love | 8.42 | 1.1082 | 25.0 | 317.0 | 328.0 | 23.0 |
| very_positive | happy | 8.30 | 0.9949 | 65.0 | 1372.0 | 1313.0 | 375.0 |
| very_positive | laughed | 8.26 | 1.1572 | 3334.0 | 3542.0 | NaN | 2332.0 |
| very_negative | terrorist | 1.30 | 0.9091 | 3576.0 | NaN | 3026.0 | NaN |
| very_negative | suicide | 1.30 | 0.8391 | 2124.0 | 4707.0 | 3319.0 | 2107.0 |
| very_negative | rape | 1.44 | 0.7866 | 3133.0 | NaN | 4115.0 | 2977.0 |
| very_negative | terrorism | 1.48 | 0.9089 | NaN | NaN | 3192.0 | NaN |
| very_negative | murder | 1.48 | 1.0150 | 2762.0 | 3110.0 | 1541.0 | 1059.0 |
| highly_contested | fucking | 4.64 | 2.9260 | 448.0 | NaN | NaN | 620.0 |
| highly_contested | fuckin | 3.86 | 2.7405 | 1077.0 | NaN | NaN | 688.0 |
| highly_contested | fucked | 3.56 | 2.7117 | 1840.0 | NaN | NaN | 904.0 |
| highly_contested | pussy | 4.80 | 2.6650 | 2019.0 | NaN | NaN | 949.0 |
| highly_contested | whiskey | 5.72 | 2.6422 | NaN | NaN | NaN | 2208.0 |
| weird_or_culturally_loaded | weekend | 8.00 | 1.2936 | 317.0 | NaN | 833.0 | 2256.0 |
| weird_or_culturally_loaded | whiskey | 5.72 | 2.6422 | NaN | NaN | NaN | 2208.0 |
| weird_or_culturally_loaded | churches | 5.70 | 2.4599 | NaN | 2281.0 | NaN | NaN |
| weird_or_culturally_loaded | capitalism | 5.16 | 2.4524 | NaN | 4648.0 | NaN | NaN |
| weird_or_culturally_loaded | porn | 4.18 | 2.4302 | 1801.0 | NaN | NaN | NaN |

Upon examination of these 20 words across four categories, it reveals how happiness scores are more than numbers, they depict cultural values, social contexts, and historical moments. 

**Very Positive Words:** The top rated are “laughter”, “happiness”, “love”, “happy”, and “laughed”. These focus on joy and human connection. More deeply these appear in almost all corpora, which can suggest that positive emotions transcend genre. Whether in casual Tweets, or Google Book’s literature, NYT media, or song lyrics, humans consistently use these words to express what matters the most. The low standard deviations (0.93-1.16) indicate the potent cultural consensus: we collectively agree these words feel good. 

**Very Negative Words:** The lowest rated words are “terrorist”, “suicide”, “rape”, “terrorism”, and “murder” which reveal the worst fear of society. The pattern of “terrorism” was primarily discussed in formal news contexts, but not casually or in songs. “Rape” appears in Twitter, NYT, and Lyrics but it doesn’t in Google Books which can possibly reflect historical censorship or the change of social will to discuss sexual violence. These absences are as meaningful as the presences. 

**Highly Contested Words:** The highest standard deviation’s words are “fucking”, “fuckin”, “fucked”, “pussy”, and “whiskey” which are all linguistic fault lines that can intensify joy such as “fucking amazing” or express aggression “fuck you”. On the other side “whiskey” appears only in the lyrics' corpus, which reveals how alcohol in songs carries dual meaning. It could mean celebration in some contexts, heartbreak in others. These words portray how context is everything. 

**Weird or Culturally Loaded Words:** weekend, whiskey, churches, capitalism, porn. This category includes words that are either culturally loaded, surprising in their scores, or carry complex social connotations that resist simple emotional classification. Unlike the clear consensus seen in very positive or very negative words, these terms reveal how cultural context, personal experience, and ideological position shape emotional response. "Weekend" scores surprisingly high (8.00), nearly matching words like "love" and "laughter," reflecting the universal human association of weekends with rest, leisure, and a rare moment of cross-cultural consensus in this otherwise contested group. In contrast, "whiskey" (5.72) and "churches" (5.70) sit near the middle but carry polarized meanings. "whiskey" can represent celebration or addiction, while churches evoke spiritual comfort for some and exclusion or historical oppression for others. "Capitalism" (5.16) and "porn" (4.18) are even more politically and morally charged—their emotional valence depends entirely on the speaker's ideology, generation, and community norms. These words demonstrate that happiness scores often mask deeper cultural battle. they are not simply positive or negative, but rather sites of contested meaning where different communities project radically different values onto the same term.

**Conclusion:** This exercise portrays that the word's happiness scores are cultural artifacts. There is a reflection of the values, fears, and disagreements of a specific time frame-  2011, and a specific population- Mechanical Turk workers. The corpus presence patterns show how words depict a different meaning across news, songs or casual conversation. A word’s happiness is a reflection about who uses it, where and why.


# Critical Reflection

## 4.1 Reconstruct the pipeline (data provenance)

The labMT 1.0 dataset was constructed through a multi-stage process that transformed raw text collections into the numerical happiness scores we've been analyzing. Based on Dodds et al. (2011) and our examination of the data structure, here is the reconstruction of how this dataset came to be:

## Step 1: Corpus Selection and Word Extraction

The researchers first assembled four distinct text corpora representing different domains of language use:

| Corpus | Source | Language Type |
|--------|--------|---------------|
| Twitter| 4.6 billion tweets (2008-2010) | Social media, informal, conversational |
| Google Books| Millions of digitized books | Formal, literary, academic, diverse genres |
| New York Times| 1.8 million articles (1987-2007) | Journalism, news reporting, formal prose |
| Lyrics| Song lyrics from various genres | Poetic, emotional, rhythmic language |

From each corpus, they extracted word frequency lists, counting how many times each word appeared. This produced four separate ranked lists showing the most common words in each text domain.

## Step 2: Creating the Master Word List

The researchers then compiled a master list of words to be rated. This wasn't simply all words from all corpora. Instead, they needed a manageable set that represented common English vocabulary. The final list contains 10,222 words, selected based on:
  - Appearing sufficiently often across multiple corpora
  - Covering a range of frequencies (from very common to moderately rare)
  - Including words with linguistic and cultural interest

This is why each corpus column has exactly 5,000 non-missing values. As each corpus contributed its top 5,000 most frequent words to the master list.

## Step 3: Happiness Rating Collection via Amazon Mechanical Turk

This is the most crucial step where raw text became emotional data. The researchers used Amazon's Mechanical Turk platform to crowdsource happiness ratings:
 - Raters: Each word was shown to 50 unique individuals (all US-based, English-speaking)
 - Task: Raters were asked "How happy does this word make you feel?"
 - Scale: 1 (sad) to 9 (happy) - a 9-point Likert scale
 - Process: Words were presented in random order, one at a time, without context

The choice of 50 raters per word represents a balance between statistical reliability and cost. With fewer raters, individual biases would have too much influence. With more raters, the cost would become prohibitive.

## Step 4: Statistical Aggregation

For each word, the researchers calculated two key metrics from the 50 ratings:

| Metric | Formula | What It Tells Us |
|--------|---------|------------------|
| happiness_average | Mean of all 50 ratings | The central tendency of emotional response |
| happiness_standard_deviation| Standard deviation of ratings | How much people disagreed about the word |

The happiness_rank column (1 = happiest word) was then computed by sorting all words by their average happiness score. This rank is what gives the dataset its name. It's a "hedonometer" or happiness meter that can rank words by emotional valence.

## Step 5: Frequency Rank Integration

Finally, the researchers integrated the frequency information from the original corpora:
 - For each word, they recorded its frequency rank in each corpus (1 = most frequent)
 - If a word didn't appear in a corpus's top 5,000, it was marked as missing (`--` in the raw data, converted to `NaN` in our analysis)

This integration created the dataset structure we've been working with: one row per word, with columns for happiness metrics and four corpus-specific frequency ranks.

## Step 6: Data Publication

The resulting dataset was published as supplementary material alongside the 2011 paper "Temporal Patterns of Happiness and Information in a Global Social Network" in PLOS ONE. The dataset includes:
 - 10,222 words
 - 8 data columns (word, happiness_rank, happiness_average, happiness_standard_deviation, and four corpus ranks)
 - Tab-separated format with metadata headers

## What This Pipeline Reveals

This generation process explains several features we observed in our analysis:

1. Missing ranks occur because a word wasn't frequent enough in a particular corpus to make its top 5000. It is not because the word doesn't exist in that domain.
2. Standard deviation measures genuine disagreement among raters, not ambiguity in the word itself.
3. The 2011 time stamp means all ratings and frequency data reflect language use from approximately 2008-2010. Words like "tweet" (rank 107 on Twitter, missing from NYT) had different meanings.
4. Cultural bias is baked in from the start. All raters were US-based English speakers, so the happiness scores reflect American emotional associations, not universal human response.

Overall, this pipeline transforms messy and context-dependent human language into clean numerical data. It is a powerful simplification, but one that comes with important limitations we'll explore in the next section.

## 4.2 Consequences and limitations

## Only high-frequency words (top 5000 per corpus)
Each source corpus only contributed its top 5,000 words by frequency. Words outside these frequency bands never enter labMT at all. The dataset focuses on mainstream, high-frequency vocabulary and largely ignores rare, technical, or niche words. This makes it easier to measure the emotional tone of “ordinary” language across large corpora but makes it hard to analyze specialized domains (e.g., medical jargon, fandom slang, minority dialects). 

For example, every rank column (twitter_rank, google_rank, nyt_rank, lyrics_rank) has exactly 5,000 non-missing values, and together they cover about 48.9% of the lexicon per corpus. Our overlap analysis shows 327 words (3.2%) that appear in none of the four top-5000 lists. Tthese words are present in labMT (because they came from at least one corpus’s 5000 list before merging), but in practice we cannot tie them strongly to any particular corpus. If we wanted to study less frequent, emerging slang or technical terms, labMT would simply not “see” them.

## Rating words in isolation, without cultural context
Mechanical Turk workers rated words alone, with no sentence or situational context. This makes the dataset easier to collect and apply (we only need word → score), but it ignores polysemy (multiple meanings) and contextual usage. Some words can be positive in one context and negative in another; rating them out of context collapses these into a single average, often hiding the underlying disagreement.

For example, we plotted happiness_average vs happiness_standard_deviation and found a set of highly “contested” words with very high standard deviation. Words like “fucking”, “pussy”, “whiskey”, “churches”, “capitalism” all have standard deviations above 2.4. “fucking” can be a hostile insult or an emphatic positive (“fucking amazing”); “pussy” mixes sexual and gendered insult meanings; “whiskey” can be associated with social drinking or addiction; “churches” and “capitalism” have strong ideological and personal connotations. The high disagreement indicates that different raters “saw” different senses of the same word—context that the dataset cannot capture.

## Using a single 1–9 “happiness” dimension
The labMT ratings reduce emotional response to a single valence dimension (1 = unhappy, 9 = happy), without measuring arousal (calm/excited), dominance (in control/overwhelmed), or more nuanced categories (e.g., nostalgia, irony). Therefore, the complex or mixed emotions are forced onto a single “happiness” line. Words that evoke ambivalent feelings (e.g., “whiskey”, “mortality”) may have mid-level averages that mask the fact that some people feel strongly positive and others strongly negative.

For example, “whiskey” has happiness_average ≈ 5.72 but happiness_standard_deviation ≈ 2.64, placing it among the most contested words. The mid-range average might tempt us to call it “neutral,” yet the high sd reveals polarized reactions.
A more multidimensional instrument could separate “pleasant excitement,” “guilty pleasure,” or “danger,” which are all collapsed here.

## Mechanical Turk as annotator population
All ratings come from workers on Amazon Mechanical Turk, primarily English-speaking internet users who opted into such tasks around 2010–2011.The emotional scores reflect the cultural and demographic biases of that annotator pool (likely overrepresenting certain countries, age groups, and internet-savvy populations).
Words tied to specific political or religious debates (e.g., “capitalism,” “churches”) will be colored by the prevailing attitudes of those workers, not by some abstract universal meaning.

For example, “churches” (avg ≈ 5.70, sd ≈ 2.46) and “capitalism” (avg ≈ 5.16, sd ≈ 2.45) show high disagreement.
These disagreements likely reflect differing personal experiences and political views among Turkers (e.g., religious vs secular, pro- vs anti-capitalist). If we applied labMT in a different cultural context (e.g., outside the U.S.), these scores might not generalize.

## Corpus selection (Twitter, Books, NYT, Lyrics) and genre bias
The lexicon is derived only from four English-language corpora. The dataset is heavily tuned to written English in particular genres, including conversational social media; published books; mainstream news and popular music. It under-represents spoken, non-digital, non-English, and non-mainstream communities. What labMT treats as “common” vocabulary is really “common in these four specific genres.”

For example, our overlap matrix shows that Google Books & NY Times are the most similar pair (3,414 words in common; 33.4% of the lexicon), while NY Times & Lyrics are the least similar (2,241 words; 21.9%). Words like rt, lol, haha, gonna, wanna are highly frequent on Twitter but do not appear in the NYT top-5000 at all. Conversely, NYT and Google Books likely share more formal, topic-specific words that are rare on Twitter or in lyrics. This means labMT is excellent for measuring sentiment in these four genres, but might miss important vocabulary in, say, scientific forums, gaming chat, or multilingual communities.

## Snapshot of language for certain time period only
The corpora and ratings reflect language usage around 2008–2011. The lexicon and ratings do not automatically update as language evolves. New slang, memes, and shifting connotations (e.g., of political terms) are not captured.

For example, words like rt, lol, blog appear as very frequent on Twitter in our 2011-era rankings. More recent slang (e.g., “yeet”, “stan”) is absent from labMT entirely. If we used labMT today without updating it, we would mis-measure or ignore large parts of current online language.

## 4.3 If you were to use this dataset as an instrument today…

The LabMT dataset is best understood as a lexical affect instrument rather than a measure of lived emotional experience. We would trust it to approximate large-scale trends in average lexical valence across corpora, especially when analyzing aggregate shifts in tone (e.g., comparing overall positivity in news versus song lyrics). Because it is standardized and reproducible, it works well for macro-level comparisons and computational modeling of sentiment trends.

More specifically, we would trust it most when analyzing high-frequency, widely shared vocabulary where annotators show strong agreement (e.g., words like “laughter” or “suicide,” which tend to produce low standard deviations). In such cases, the dataset provides relatively stable estimates of collective emotional valence. It is also appropriate for studying long-term changes in average tone across large text collections over time, provided that the texts resemble the source corpora on which the lexicon was built.

However, we would refuse to claim that it captures “true emotion” or contextual meaning. The dataset assigns a single scalar value to words presented in isolation, ignoring irony, sarcasm, genre, identity, and pragmatic use. Our disagreement analysis showed that words such as fucking, whiskey, and capitalism produce high standard deviation scores, indicating that affect depends heavily on interpretation. Therefore, LabMT should not be used to draw conclusions about speaker intention, community identity, moral stance, or individual emotional states inferred from word usage.

We would also refuse to generalize its scores as universal judgments. The ratings reflect a specific annotator population and a particular cultural moment (early 2010s English-speaking participants). Rare words, emerging slang, and non-English terms fall outside its reliable scope. In addition, mid-range averages for highly contested words should not be interpreted without consulting their standard deviations, as an apparently “neutral” average may mask polarized reactions.

If we were to rebuild this instrument today, we would introduce several improvements. First, we would collect contextualized ratings (short sentence fragments rather than isolated words) to better handle negation, irony, and multiword meaning. Second, we would diversify and systematically document the rater pool across regions, age groups, and linguistic backgrounds, recording demographic metadata to make bias visible rather than implicit. Third, we would update and expand the source corpora to include newer platforms (e.g., online forums, social media, spoken transcripts) and potentially multiple languages. Finally, we would move beyond a single “happiness” dimension toward a multidimensional affect model (e.g., valence, arousal, dominance, or discrete categories such as anger, fear, and joy).

These changes would make the dataset more sensitive to ambiguity, social variation, and contextual nuance while retaining its usefulness for large-scale computational analysis.


# How to Run the Code

## Structure layout 

 - `src/` — Python analysis scripts
 - `data/raw/` — input data (E.g Data_Set_1.txt)
 - `data/processed/` - cleaned dataset used for analysis
 - `figures/` — PNG plots
 - `tables/` — CSV tables/summaries
 - README.md
 - requirements.txt


## Setup Steps 

 1. Clone the repository
git clone https://github.com/auroraliu0312/labMT-hedonometer-project

 2. Create and activate virtual environment
 python3 -m venv .venv
 source .venv/bin/activate  # On Mac/Linux
 or .venv\Scripts\activate  # On Windows

 3. Install dependencies
 pip install -r requirements.txt

 4. Run the analysis
 python3 src/data_analysis.py

 5. What gets generated?
 After running, look in:
- `figures/` — PNG plots
- `tables/` — CSV summary tables

# Credits

## Team roles:
1. Repo & workflow lead: Anny Li
2. Data wrangler: Mohan Liu
3. Quantitative analyst: Mohan Liu, Anny Li
4. Qualitative / close-reading lead: Angelina Roman Rosales
5. Provenance & critique lead: Simone van Moerkerk
6. Editor & figure curator: Jaena Danaram

## Citation of papers:
Dodds, Peter Sheridan, Kameron Decker Harris, Isabel M. Kloumann, Catherine A. Bliss, and Christopher M. Danforth. 2011. “Temporal Patterns of Happiness and Information in a Global Social Network: Hedonometrics and Twitter.” Edited by Johan Bollen. PLoS ONE 6 (12): e26752. https://doi.org/10.1371/journal.pone.0026752.

## Academic integrity & AI note
During the code construction process, we made limited use of AI-based tools for support purposes.

In the early stages of development, we consulted DeepSeek to help debug code and clarify specific technical questions. For parts of the Results section, we used ChatGPT to refine phrasing, improve clarity, and structure initial drafts of explanations. Throughout both the drafting and revision stages, we also used the UvA AI assistant to review wording, check coherence, and strengthen academic tone.

All code included in the repository was revised and verified by us. We understand the logic and functionality of each script and are able to explain the analytical steps, statistical calculations, and design choices in detail. AI tools were used as writing and debugging support rather than as a substitute for conceptual understanding or interpretive reasoning.

Additionally, all interpretive claims, methodological decisions, and critical reflections represent our own academic judgment and responsibility.


## Mini-Project 2: Met Museum Happiness Analysis

### Data Acquisition

## Mini-Project 2: Met Museum Happiness Analysis

### Data Acquisition

**Role: Data Acquisition Lead**  
*Responsible for API integration, data fetching, and provenance documentation*

---

####  Acquisition Pipeline

**Step 1: Search Strategy**
- **API Endpoint**: `https://collectionapi.metmuseum.org/public/collection/v1/search`
- **Search Terms**: 18 emotional keywords (love, death, war, peace, nature, beauty, sorrow, joy, flowers, landscape, portrait, religious, happiness, sadness, victory, defeat, celebration, mourning)
- **Parameters**: `hasImages=true` (limits to objects with available images)
- **Rate Limiting**: 0.3s delay between searches (respects API guidelines)
- **Logged Parameters**: All search terms and timestamps recorded in script output

**Step 2: Object Collection**
- For each keyword, collected first 15 object IDs
- Total unique objects targeted: ~270
- Saved raw ID list to: `data/raw/met/met_object_ids.csv`

**Step 3: Metadata Fetching**
- **API Endpoint**: `https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}`
- **Fields Extracted**: title, department, classification, culture, period, dates, medium, artist details, accession year, tags
- **Rate Limiting**: 0.2s delay between requests
- **Output**: `data/raw/met/met_artworks_raw.csv`

**Step 4: Processing**
- Removed entries without titles
- Created `century` field from `object_begin` for temporal analysis
- Cleaned titles (removed punctuation) for hedonometer scoring
- **Output**: `data/processed/met_artworks_processed.csv`

**Fetch Script**: `src/met_fetch.py`

**Sample Raw Data (First 3 Rows):**object_id,title,department,classification,culture,object_date,object_begin
437373,"The Death of Socrates",European Paintings,Paintings,French,1787,1787
437123,"Love and Psyche",European Paintings,Paintings,French,1798,1798
548273,"War and Peace",Modern Art,Sculpture,American,1952,1952 
---

#### Provenance Statement

The dataset comprises approximately 250-300 artworks from the Metropolitan Museum of Art's permanent collection, accessed via their public API in March 2025. The Met's collection spans 5,000+ years of world culture, from ancient Egyptian artifacts to contemporary art, with particular strengths in European painting, American art, and Asian art. However, this dataset does **not** represent a random or comprehensive sample of the Met's collection. Rather, it is a **keyword-convenience sample** selected using emotionally charged search terms (love, death, war, peace, etc.). This introduces significant selection bias: the dataset overrepresents artworks with emotionally expressive titles and underrepresents those with descriptive, abstract, or non-English titles. The collection itself reflects historical collecting practices that have favored Western European art, meaning non-Western cultures are underrepresented relative to their global significance. Users should interpret findings as reflecting "artworks with emotionally resonant English titles in the Met's collection" rather than "art" or "human expression" broadly.

---

####  Ethics Note

**Privacy & Consent:** All data collected is from the Met's public API, which provides information about artworks in the museum's permanent collection. No personal data about museum visitors, donors, or staff was collected. Artists represented are historical figures whose information is publicly documented; no living artists were identified or contacted.

**Platform Constraints:** The Met API is freely available for educational and research purposes. Usage respects the platform's rate limits (implemented delays of 0.3-0.5 seconds between requests) and terms of service. The API does not require authentication, and no API keys are stored in the repository.

**Representation & Bias:** This dataset inherits the biases of the Met's collection itself, which reflects over a century of Western collecting priorities. European art is overrepresented; African, Oceanic, and Indigenous American art are underrepresented. Additionally, the keyword search strategy favors artworks with English-language titles, excluding works cataloged primarily in other languages. This limits the cultural scope of any findings to Anglophone interpretations of art.

**Limitations:**
- Cannot generalize findings to "all art" or "human creativity"
- Time periods are unevenly sampled (Renaissance/Baroque overrepresented)
- Only objects with digital images are included (may exclude important works)
- Artist demographic data (gender, nationality) is incomplete in source API
- Titles may be translations, not original language

**Responsible Use:** This dataset should be used to explore questions about how emotionally charged language appears in artwork titles within a major Western museum collection. It should **not** be used to make claims about:
- Cross-cultural differences in emotional expression
- Historical trends in art making (only titles, not artworks themselves)
- Artist intentions or audience reception

---

####  Data Dictionary: `met_artworks_processed.csv`

| Column | Type | Description | Missing Values |
|--------|------|-------------|----------------|
| `object_id` | int | Unique Met object identifier (primary key) | 0 |
| `title` | string | Original artwork title as cataloged | 0 |
| `department` | string | Met curatorial department (e.g., "European Paintings") | 45 |
| `classification` | string | Type of object ("Paintings", "Sculpture", etc.) | 67 |
| `culture` | string | Cultural origin ("Italian", "French", "Japanese") | 89 |
| `period` | string | Historical period ("Renaissance", "Edo period") | 156 |
| `object_date` | string | Original date inscription (may be non-numeric) | 23 |
| `object_begin` | int | Numerical start date for chronological sorting | 34 |
| `object_end` | int | Numerical end date | 34 |
| `medium` | string | Materials and technique | 78 |
| `artist_name` | string | Creator name(s) | 112 |
| `artist_nationality` | string | Artist's cultural/national identity | 145 |
| `artist_begin` | int | Artist birth year | 167 |
| `artist_end` | int | Artist death year | 167 |
| `accession_year` | int | Year museum acquired the work | 89 |
| `century` | int | Century derived from object_begin (e.g., 1800 for 19th C) | 34 |
| `title_clean` | string | Title with punctuation removed (ready for hedonometer) | 0 |
| `tags` | string | JSON list of subject tags from API | 201 |

**Missingness Notes:**
- Artist information is often missing for anonymous works or non-Western art
- Culture field may be oversimplified ("Italian" instead of "Florentine")
- Dates are approximate for ancient/medieval works (indicated by "±" in original)

---

####  How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/auroraliu0312/labMT-hedonometer-project.git
cd labMT-hedonometer-project

# 2. Install dependencies
pip install requests pandas

# 3. Run the fetch script
python src/met_fetch.py






