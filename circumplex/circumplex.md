# Affective Dimensions of Word Embeddings

Use embeddings to do data science on words.

## Circumplex of Affects

In 1980, psychologist John Russell asked 36 university students to categorize how 28 particular affects -- a term used by psychologists encompassing feelings, moods, and emotions -- made them feel ([Russel 1980](https://www.researchgate.net/publication/235361517_A_Circumplex_Model_of_Affect)). In analyzing the results he found that the responses were consistent with the theoretical _circumplex model of affect_:

<OG CIRCUMPLEX>

Previous studies told Russell that the axes of the circumplex captured a significant portion of how people distinguished between affects. The horizontal axis is _valence_, which describes if an affective experience is positive or negative. The vertical axis is _arousal_: here, that's a subjective measure of the energizing or pacifying effect of an affect, but in other studies it's measured as the relative activity of the sypmathetic nervous system.

What's amazing about this model is that it's been independently verified several times ([Posner et al. 2005](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2367156/) has a good overview on the empirical evidence) and seems to hold across languages and cultures ([Russel 1983](https://web.archive.org/web/20170809041458id_/https://www2.bc.edu/james-russell/publications/pancultural%20aspects.pdf), [Loizou and Karageorghis 2015](https://www.tandfonline.com/doi/full/10.1080/1612197X.2015.1039693)). So while all models are wrong, _this_ model at least seems to be consistent with how the average person conceptualizes emotional states. And that's true even if they're not consciously aware of the fact that they have a conceptualization at all!

Which makes me wonder: if _valence_ and _arousal_ are such consistent and universal concepts, surely they're implicit in our writing. Have large language models (LLMs) like ChatGPT picked up on the pattern? That's not to say that ChatGPT experiences affective states -- you can't understand the experience of arousal without a sympathetic nervous system -- but the underlying model might have a latent representation consistent with ours.

Let's find out.

Before we do, I want to clarify the point of this experiment. First, I'm not interested in what ChatGPT has to _say_ about affective states: if you ask you'll get definitions and assurances that the model doesn't "feel" (although it also told me that being helpful sometimes means matching emotional tone).

Second, there are _absolutely_ emotional patterns in natural language to be found. Computer-aided analysis of text goes back to the 60s ([Stone et al. 1966](https://mitpress.mit.edu/9780262690119/the-general-inquirer/)), and in the early 2000s many researchers applied similar techniques to predict the emotional content of text ([Turney 2002](https://arxiv.org/pdf/cs/0212032), [Pang et al. 2002](https://www.cs.cornell.edu/home/llee/papers/sentiment.pdf), [Snyder and Barzilay 2007](https://aclanthology.org/N07-1038.pdf)). But modern LLMs aren't explicitly trained on these patterns.

__What I want to know__ is if LLMs, while being trained to predict patterns in text and later fine-tuned for things like "being helpful", have somehow encoded valence and arousal in their latent states.

To see if that's the case we'll look at _word embeddings_, which map bits of text to the state that LLMs use to compute what to say next. Embedding models are often trained jointly with LLMs, but word embeddings themselves pre-date the current wave of LLM-based AI research by several decades. The `word2vec` model ([Mikolov et al. 2013](https://arxiv.org/pdf/1301.3781)) made headlines when it was published in 2013 because it let us use _math_ on words to do things like make analogies. For instance, "_Einstein : scientist :: Messi : ?_" could be solved by calculating `scientist - Einstein + Messi` and finding the nearest vector -- in this case, `midfielder`. Mind-blowing for the time.

For this experiment I'll use OpenAI's `text-embedding-3-small` model to embed Russell's 28 affects (it's cheap and fast). We'll see if there's any relation with the circumplex model of affect, and if there is we'll train a simple model to predict the valence and arousal of _other_ words.

One problem: Russell only includes precise measurements for a handful of the 28 affects. To get the rest, I imported the figure into some photo-editing software and measure the angles myself. My results matched those Russell reports to within a degree -- perhaps surprising, considering the imprecision (or maybe it's just _character_) of what is clearly a hand-drawn figure.

Here's the cleaned-up version of the circumplex. Hover over any of the data points to see more information.

<CIRCUMPLEX>

## Graphical Gut Check

Our first step is to look at the data and see if word embeddings have anything to do with the circumplex. And if we apply the lesson of Anscombe's quartet the best way to look at the data is to _look_ at it.

But word embeddings are way too high-dimensional to graph. Even using our small embedding model converts each word to a vector in a space with 1,536 dimensions. If your screen is anything like mine, you've only got two meaningful dimensions to work with, so we need a way to distill those 1,000+ dimensions down to two without losing too much information. One way to do that is _manifold learning_.

A manifold is anything that "looks like" Euclidean space. If we can find a two-dimensional manifold embedded in our 1,536-dimensional space that touches all our data, we can throw away those other dimensions and look at the manifold directly.

We'll be using a technique called _t-distributed stochastic neighbor embedding_ (or t-SNE) from [van der Maaten and Hinton](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) to get a manifold that we can graph. t-SNE does what it can to keep nearby vectors close in the two-dimensional projection. There's no meaningful interpretation to the precise distances -- we're just looking for any structure that might emerge.

(If you want to build more intuition on how to interpret the results of t-SNE, [Wattenberg et al.](https://distill.pub/2016/misread-tsne/) have a set of fantastic interactive examples.)

<TSNE>

Because we care about the affective dimensions, I've marked every point with an arrow whose angle matches the emotion's position in the circumplex. It looks like the arrows define a smooth field (with a few discontinuities). That's a good sign for us -- the t-SNE algorithm doesn't know anything about valence or arousal, so this structure is a strong indication that the embeddings carry some amount of information about the affective dimensions.

## Predicting Affective Dimensions

Now that we're convinced the embeddings contain some amount of the signal we want, we need a mechanism to extract it. 

Now that we've convinced ourself that the signal is there, we just have to find a way to extract it. We'll try the simplest thing first, which is a linear model. Turns out we can do quite well with a linear model. One big problems is the DoF -- a naive linear model would have 1,536 parameters to explain 28 data points. That's too flexible, we'll end up overfitting.

We could reduce the dimensionality of the data like we did with t-SNE. Unfortunately, our implementaiton doesn't let us project _new_ words, so we'd just be learning things baout the 28 data points we already have. Other standard tools like principle component analysis (PCA) reduce the dimensionality by finding redundant components of the embedding, but in my experience the information in these embeddings are pretty well distributed.

Another option is _regularization_ -- you constrain the linear model so that it isn't as flexible and shouldn't overfit. Two common regularization techniques are Lasso -- which tries to make most of the parameters 0 -- and Ridge -- which tries to keep the parameters small. Since embeddings widely distribute their content, we'll stick with Ridge.

<RIDGE>

I trained two Ridge regression models -- one for valence, one for arousal. They're supposed to be independent. We do okay on the 28 data points, although valence does considerably better than arousal. Click on any data point to see where the original was.

## Applying what we've learned

So we have a model that poorly tells us what we already knew. The fun of learning these is that we now have a pipeline -- text-embedding-3-small -> Ridge models -- that can be applied to _any_ bit of text.

Here's a bunch more affects taken from the Hoffman Institute. They're categorized roughly at the level of Russell's affects, but are otherwise quite a bit more nuanced. If we apply our same pipeline as before we end up with:

<HOFFMAN>

Click the groups on the right to focus and hover over points to see more information.

There's a lot more in the center than I might expect. That might be because the affects are actually neutral in both, or that we just don't have a strong signal for that word. Since the circumplex is really just talking about relative dimensions I think it's probably fine.

Our model clearly isn't _that_ good, but it seems to get broad trends. I don't have any ground truth for these words, so we can't compute any objective metrics of success, but feel free to poke around and see what you think for yourself.

If we're feeling spciy we can even apply this model to things that aren't affects.

## COnclusion

Not much data, but there's clearly some signal. Valence seems better captured than arousal -- I feel like that makes sense, it's a more readily available and consistent definition. Certianly more sophisticated techniques could be applied (PROBING), but we need way more data.

More to be done with embeddings. This was just words, but we could jointly encode definitions or poetry or brain scans and maybe get a better sense of what it means to experience these affects.

Concept extraction -- [OpenAI 2024](https://openai.com/index/extracting-concepts-from-gpt-4/), [Anthropic 2024](https://www.anthropic.com/news/mapping-mind-language-model)
