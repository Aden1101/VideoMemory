HANNA HAJISHIRZI: OK, let's get it started. It's my pleasure to introduce you, son, as our colloquium
0:07
speaker today. So Yu Sun is a postdoctoral researcher at Stanford
0:13
and a research scientist at NVIDIA. He's working with some of our really good friends
0:19
who we know some of them, including Carlos Guestrin. So today you will talk about some of his work
0:30
on test-time training. So, in fact, we know he's one of the first people who
0:36
started this topic called test-time training. And I've just learned you guys coined the term.
0:43
He's done this during his PhD in building these models
0:48
for computer vision applications and, most recently, extending it to many other domains
0:54
like scientific discovery, coding, and more.
0:59
And he started this in 2019 and since then it
1:04
has become very popular because its resembles the way that humans learn.
1:10
So very excited to learn about this. Thank you. YU SUN: Well, thank you for the introduction, Hanna.
1:17
So-- well, let me begin.
1:22
So a little bit about myself. So, so I grew up in the Shandong Province of China.
1:28
That's this thing over here. It looks like a rotated Like button.
1:33
It's the hometown s Confucius, he's a famous Chinese philosopher and pedagogue.
1:40
And perhaps for that reason, our province is very serious about tests or exams.
1:48
And, in fact, it is so serious that when we have exams,
1:53
we have armed police guarding the exams shipped over here in those crates being transported.
2:02
And because just-- because how carefully we
2:07
regard these things, the process of studying often
2:12
becomes a constant process of guessing the test problem. So the American kids are guessing
2:18
about betting on baseball games, we are betting on what exam problem is going to show up in the next test.
2:26
And the process of testing is really praying. As I'm going through my test printout,
2:34
I'm always praying there's a next one-- it's got to be something similar to what I have studied because, otherwise, I don't what
2:41
to do about it. And then I came to America when I was 13.
2:48
And I discovered a great thing in America.
2:55
It's called take-home test. And the great thing about take-home test is that you can learn the relevant content
3:03
while taking the test. So there is really no guesswork.
3:09
Now, just to give you an example, when I was taking machine learning theory,
3:15
actually the take-home test was given to me during the first week of class. I just have to solve a list of problems, up to 10 stars.
3:23
And, in fact, one among the list was this open problem in learning theory.
3:29
And if I just did a single problem, I could have gotten an A-plus in the course.
3:35
I could have done that in the first week of class. That was the spirit of American take-home test.
3:41
And now you ask, why would America survive this way?
3:46
Something like this-- I mean, in America, we call it take-home test,
3:51
but, in China, it's called cheating. Why would you-- why do you survive? Well, it's really because at test, at least good ones,
4:00
should be a proxy of the real world. And in the real world, we end up learning on the job
4:09
all the time. And that's perhaps speaking a language.
4:15
When I first came here, I wasn't so good. Now, after just years and years of practice on the job,
4:21
I'm better. And think about driving or perhaps the ultimate form,
4:28
doing research. So when you do research, you always
4:33
are learning on the job about your research problem. And here's an extreme case.
4:40
So this gentleman, Andrew Wiles, he spent nine years proving Fermat's Last Theorem
4:46
and succeeded. Now, when he was 10 years old, that's when he first learned about this problem and decided to be a mathematician.
4:52
And then after he became a professor, he decided to work on this problem in complete isolation.
4:59
He did that for the next seven years. And then for two more years he ended up proving the problem
5:06
and this is the big take-home test of his life, which
5:14
he did well on. Now, let's think about how testing
5:19
works in machine learning. So when this project started in 2019 machine learning
5:26
has three stages already. So we do pre-training on internet scale, general knowledge, fine-tuning on some specific domain,
5:36
and then comes testing, where you fix our model and it just gets deployed.
5:42
It's the most boring stage. Today, fine-tuning is now called post-training but not
5:51
much else has changed besides the dramatic difference in scale. Conceptually it's the same thing.
5:58
Specifically, the process of post-training is always about guessing the test instance.
6:05
And the process of testing, you're just passively praying that whatever test instance you see
6:13
is going to be something you've seen similar to what you have post-trained on. It's the same thing.
6:19
And the-- this is the case because really most machine learning was happening before you see the test instance.
6:30
The moment when the test instance arrives is really a defining moment in machine learning.
6:36
And after this test instance arrives, well, you're just passively testing there, not
6:44
much interesting happening. And I believe that, in the future, most machine learning will be happening in this part,
6:53
at that side of the boundary. Now, you might be thinking, how could it happen in this part?
6:59
I mean, testing your model is fixed. Well, the whole point is that we should be training the model at test time.
7:07
And that's what this talk will be about.
7:12
So let's zoom into this future. And the simple setting is when you
7:19
have a test problem such as this problem in machine learning theory and then you're also
7:25
given a pile of lecture notes as data. It's data given to you at test time.
7:31
And here the most fundamental problem--
7:37
and here because the test problem-- we think that it will be solved once you understand the relevant data.
7:44
So in this setting the fundamental problem is how to efficiently process a lot of data given at test time.
7:51
That's a simple setting. Now, you might think, what's the hard setting? Well, that's when you're not given the data anymore.
7:59
And this could happen because maybe your test problem is just
8:06
an open problem. There is no lecture notes that would enable you to solve this problem.
8:12
So that's what Andrew Wiles was doing in his research.
8:18
And because he could do it, we thought maybe we could formalize this process and take a crack.
8:26
So that's the outline for the talk today. We'll go through the simple setting and the hard setting,
8:32
just those two things. All right, now let's zoom into the simple setting,
8:39
And just as a reminder, you're given the data and in addition to the test problem, which is solvable
8:45
once you understand the data. So here the focus is on the data, how to efficiently process a lot of data given to you
8:54
at test time. Now data given at test time--
8:59
in machine learning, we have a special word for it. It's called context.
9:04
And if you have a lot of it we call it long context.
9:11
That's a long context problem. And people say this is among the most economically valuable
9:16
problems today because people want to give their language models large legal cases, entire code bases, and long histories
9:24
of interactions. So all the frontier labs, they have dedicated teams working on this problem, yet people also say
9:31
it's one of the biggest bottlenecks in AI today. Now, how does this become a bottleneck?
9:39
Because, after all, when you have data at training time,
9:45
you how to deal with it. You just do gradient descent on the model.
9:50
You know how to deal with a lot of data given a training time. Now, how come that at test time, when you give it
9:56
a different name, you just all of a sudden don't how to deal with it? Well, it's really because that at test time, at least
10:03
in the old days of machine learning, people weren't comfortable changing the weights of the model. So you have to come up with workarounds
10:11
to use data without changing the weights of the model. And, the most popular workaround is called self-attention.
10:20
That's the key component to transformers. So just a little bit about how self-attention works.
10:26
So in transformers, using self-attention, you want to store the embeddings of every word in a growing list
10:34
because that's how you will use the data, how you will use the words in the data. And this list is usually called the KV cache.
10:42
So here's an example. Let's say this is your data given
10:47
at test time, your context, and you want to process each of them into an embedding.
10:53
So suppose I have already processed two of those words. And now for the third one, I first do an initial projection
11:03
into embedding but now I want to incorporate the information of the words before it.
11:08
So how do I do that? Well, I just compare these embeddings with this word
11:15
and then collect all the information that I get from these comparisons.
11:21
So this process takes three units of compute in this example.
11:28
Now the important thing is that for the next embedding, this same process will take four units of compute
11:36
and the next one five units of compute. So the key idea is that now the cost per word
11:45
grows linearly with longer context. So you can see this with those empirical measurements
11:54
here that as context length grows longer, the latency, which is the cost per word, is growing.
12:03
So your model gets slower and slower for each word as it's going through your context.
12:10
And this is how long context becomes a bottleneck. So people come up with more workarounds
12:21
on top of this workaround and this one is called sliding window. So the main idea, very simple, is what if we only
12:29
keep tab of the most recent words instead of all the words?
12:35
So now let's say the most recent three words. So now I only cost three units of compute, a constant amount
12:44
no matter how long my context is. And as you can see in the empirical measurements here,
12:51
the line is flat if you have a sliding window. But this comes at a cost.
12:57
You will forget the tokens that are outside of your window. So let me help you understand this plot.
13:06
It's a very important plot. So as you have longer and longer context. I'm narrowing the language modeling loss.
13:14
And in order to clearly show the trend, I'm benchmarking the loss of everyone
13:20
against that of the orange line. So I'm subtracting the orange line from everybody. That's why the orange line itself is now becoming 0,
13:28
flat line is 0. And now you can clearly see the trend that the green line is rising steadily
13:36
compared to the orange line. So sliding window gets worse loss in longer context compared to transformers.
13:43
Not surprising. That's the price you pay for only using the most recent tokens.
13:50
And the holy grail in long context language modeling, if you want to scale this context lens,
13:56
is to be flat in both panels. So a lot of people have tried.
14:04
This has been an area with a lot of research, and I'm just picking two representative examples here.
14:12
So one is called, one is Mamba and the other is called Gated DeltaNet.
14:17
So you can see that both of them get better offset compared to sliding window
14:22
but pretty much the same scaling. So how would we solve this problem?
14:30
Now, let's just zoom back for a moment
14:35
and think about how this problem showed up in the first place. So it happened in the first place
14:41
because we at test time, when you have data or context,
14:48
people aren't comfortable with just changing the weights of the model. I mean, we aren't comfortable with just
14:54
taking the approach at this side of the boundary and doing it over here.
15:00
So what if you just do that, you just forget about the boundary
15:05
and use the same approach? And that was the main idea of my paper back in 2024.
15:12
And since then, there have been a lot of direct follow-ups. And this line of work has heavily
15:18
influenced the modern revival of continual learning. And today I will talk about my most recent work, which
15:24
is really the simplest of all. So suppose we have a long sequence of words, that's
15:35
your context, and you only want to use a transformer with sliding window.
15:40
That's the green thing, which is supposed to have constant latency across time. And now you not only do a forward pass for this model,
15:51
you also do a backward pass using the next token prediction loss, which is just the same loss as you
15:59
use during pre-training. And this enables you to compress the content inside this window
16:09
into the weights of this model as it's moving outside of that window.
16:16
So right now while the model could only attend to the context that's immediately below it,
16:23
the context before it has been compressed into the weights of the model. And over here you do the same thing and so on and so forth.
16:33
It was very easy. So after all these steps of gradient descent,
16:40
when your model is finally over here, even though most of the context is outside of this window,
16:46
they should have been-- the information in that context should have been in the weights of the model
16:52
because you are doing backprop to update those weights. That's the main idea.
16:58
And contrast this with just the transformer with self-attention over the entire context window, over the entire sequence,
17:07
which is just going to do a giant arrow of forward pass. And the question is, are all these little arrows
17:14
combined going to be faster than this one big forward arrow?
17:22
And are they going to be as effective?
17:27
Are they going to compress the content outside of the window as I said?
17:33
So, for this, let's look at the results. So just as a reminder, there's a holy grail here,
17:39
is to be flat in both of these panels. And here you want to have constant latency.
17:47
And over here you want to have the same loss as transformers across context lens.
17:53
First of all, for latency, well, this is how our method does. So it is indeed slower than sliding window
18:01
because you have to pay the cost of test-time training. You have to do the backward pass.
18:09
And so that's indeed slower. But, at around 32,000 and afterwards,
18:14
that's when it is actually faster than the giant arrow--
18:19
the giant forward arrow of a transformer that attending to the entire context.
18:27
And that's for prefill at 128k, is 2.7 times faster in prefill
18:33
and it will be six times faster for decode. So in order to make it work for this side of the plot,
18:43
there are some important details. So I was using the analogy of testing and take-home tests.
18:52
So in regular machine learning, as I was saying, at test time, we want to do well on these tests.
18:59
So training is like doing practice tests. Now, I have just changed the process of testing
19:06
to the analogy of take home tests, because we're doing test-time training. But take-home tests-- doing well on take-home test
19:14
is its own kind of skill. And that skill can be pretty different from doing
19:22
well just on regular practice, just on regular tests. So if you just practice regular tests during training,
19:28
you're practicing the wrong skill. So in order to practice the right skill, you want to also do a practice take home tests, which
19:36
in the language of machine learning, we call it learning to learn. Also called meta-learning.
19:41
So this is an important detail that we need to take care of. And once you do that, you see that our method here is indeed
19:52
flat compared to transformers. In fact, it has a constant advantage. So this gap means that it's doing better than transformers
20:00
for all context lengths. And this is the first method with this scaling behavior
20:09
to the best of my knowledge. So if you put back all the other methods that have constant
20:15
latency, you'll see that, yeah, because of the different scaling on that panel of the plot, there is a big difference for 128,000
20:25
context. AUDIENCE: [INAUDIBLE] YU SUN: Yes. AUDIENCE: What exactly are you doing under the heading
20:31
of meta-learning? YU SUN: Under the heading of? AUDIENCE: Meta-learning. YU SUN: Of meta-learning?
20:36
AUDIENCE: Yeah. YU SUN: Yeah, so the most naive approach is to say, OK, so I'm going to take each training sequence,
20:50
do gradient steps on it, go through the entire sequence, which could be a lot of gradient steps,
20:56
and collect the language modeling loss over this sequence.
21:02
Now, that loss rate is exactly the loss you care about.
21:08
So during training time you could backprop through that loss.
21:14
And because the process of generating that loss already includes taking gradient steps, the process
21:21
of backprop through that loss would mean that you need to take gradients of gradients.
21:26
So, concretely, this optimization would involve gradients of gradients.
21:33
Now, this could be slow. So there are good first order approximations
21:40
to these gradients of gradients. Namely you don't actually need to do them, but that's a conceptual way to think about it.
21:49
Did I answer your question? AUDIENCE: Not en-- let me ask the question another way. What goes wrong if you just do backprop at test time?
21:59
YU SUN: And at training time you don't modify any of the pre-training algorithm, right?
22:06
AUDIENCE: Now you have your model and then you have your test time data and you just do backprop as before. That's the easiest.
22:12
What goes wrong? YU SUN: Yes, so that is what we do. The only thing that we change in that important detail part
22:20
is how we prepare for test-time training is what we do before test time at regular training time.
22:28
Did I make sense? AUDIENCE: I understand, but I'll ask you more about it when you-- YU SUN: Sounds good.
22:33
Any other question? This is a good point to be asking questions.
22:39
This finishes the first little part of my talk. Yes, please. AUDIENCE: How large is the context window for your method?
22:47
YU SUN: Yeah, so it's 8,000. That's why you'll see is that--
22:52
I mean they're just taking the green line basically. We're taking the green line as a baseline and doing test-time training on top of that.
22:59
So that's why you see that the green line and the orange line are the same over here. AUDIENCE: I see. And have you experimented with making it smaller
23:06
and seeing how small it can go before you get higher loss?
23:11
Because it seems like it's a different kind of assimilation of context.
23:18
YU SUN: Yeah, we have done smaller, but it's not quite that interesting, I
23:24
feel, because, I mean, smaller, you don't go much faster.
23:30
Basically 8,000, it saturates the hardware.
23:35
And if it's 4,000, you don't get much of a speed up. AUDIENCE: Right. YU SUN: Yeah.
23:41
AUDIENCE: I just wonder if you can make it very short and-- YU SUN: I have gone down to 1, 0--
23:48
depending on how you think about it. Basically the limiting word would just be you don't have attention at all.
23:57
It's like a bigram. If it you-- if you say something like,
24:04
I don't know, Stanford University. It's going to work because you will see Stanford and predict University. It's a two-word association.
24:10
If you give it University of Washington, it will see "of" and not what to say next because it's not
24:17
a two-word association. I've tried it with that model. Yeah, in-- for that model, the improvement
24:25
of doing test-time training is, of course dramatic. But you will lose some performance
24:31
compared to the performance transformer with self-attention. And I feel that loss is rather unnecessary because that model
24:38
isn't so much faster actually. Yes. AUDIENCE: On the other hand, what
24:45
happens if I don't care about runtime, if I just take a really large context and I use meta-learning, do I get substantially better loss performance?
24:54
YU SUN: So you are saying that you take the orange line
25:01
and do test time-training on top of that. AUDIENCE: Yeah. YU SUN: Yes, you get better performance.
25:06
AUDIENCE: How much better? YU SUN: It is better than-- it is going to be better than the blue line but not quite-- not a huge--
25:14
something, well, I believe it will be something like-- so this is something like 0.02,
25:22
or 0.016, or something. Well, it will probably double this disadvantage.
25:27
It is in the paper. Yeah, it is an experiment in the paper. Just check out the paper.
25:39
Anything else? All right, if not-- yes, please. AUDIENCE: How are you evaluating loss here during inference time?
25:51
YU SUN: So it is just language modeling loss. So how do you normally evaluate language modeling loss?
25:58
AUDIENCE: --next token prediction. YU SUN: That's right. AUDIENCE: Like if you're interacting with a chatbot and you want it to learn the context of your conversation
26:07
what is the-- OK, I think I see what the loss means.
26:15
YU SUN: All right, sounds good. If there's no more questions, then I will go on.
26:22
Let me go on then. Cool, so I have covered the simple setting.
26:28
Now, what about the hard setting? That's when you're not given the data.
26:35
And how do you do that? So just to say it in a different way, in the simple setting,
26:42
your test instance is solvable once you have learned from the context.
26:48
But in the hard setting, you're not given this bridge. So how do you build a bridge of that sort?
26:55
And in machine learning, well, this can come up, by the way, when your test instance is
27:01
an open research problem. And in machine learning, our abstraction for this is called out-of-distribution generalization.
27:07
So that's when your test instance is not represented by the training data.
27:13
So back in 2018, when this line of research first started,
27:18
I was studying this problem in the context of object recognition. So say you have ImageNet over here and you train on that,
27:27
but you want to run your model on a snowy day and you have never seen a snowy day in your training set.
27:35
What do you do? So, first, you notice that if you
27:40
are to train on regular ImageNet and test on a snowy day, of course, there is a dramatic performance
27:45
drop, even for the state of the art models. And if you train and test both on snowy day,
27:52
if you were able to do that to eliminate the distribution shift, yeah, since there is no drop.
27:59
It's this distribution shift that's causing the trouble. So wouldn't it be great if we had
28:06
training data on a snowy day? Well, in fact, we do, it is just on this side of the boundary.
28:16
And it is a very test instance itself that can serve as a piece of training data on a snowy day.
28:25
You just need to be willing to train on it. So the central difficulty is that this test input
28:32
doesn't come with it's label. That's why it's called a test input. So how to generate data from an unlabeled
28:40
input for test-time training? That was a difficult question back then.
28:47
It still is today. So back then my answer was to use self-supervised learning.
28:53
Let's see what that means. So say you're doing object recognition and you have this neural network and it's giving
29:00
the wrong prediction right. Because it's deep learning, this sort of prediction is usually wrong because the network
29:07
has bad intermediate features. That's what will break deep learning.
29:14
And self-supervised learning is a way to fix the features with only unlabeled labeled input, exactly
29:25
what we want. So let me explain it using an example and let me make some space over here.
29:30
So this example is called masked reconstruction.
29:38
So you have this image and you can just apply a random mask to it and ask the neural network
29:44
to reconstruct it. For that task to type check, you have to give it a different head.
29:50
That's the reconstruction head and that's going to be different from the main task head up there.
29:57
But as you try to reconstruct this image,
30:02
if you ask the neural network to form, it will have to, I mean, form good intermediate global representations for the image
30:11
in order to fill in the masked parts. So the process of training is just
30:18
to learn how to better reconstruct by comparing the original image to the reconstructed one
30:24
and you keep doing this comparison. And in the end of training you, once you're
30:31
able to reconstruct the image file, you would have learned the good features.
30:36
So now you're ready to put back the main task head and make a prediction.
30:44
And if it doesn't succeed, you just repeat this process again with a different random mask.
30:49
So this was the result we got back in 2019. As you take more gradient steps at test time
30:58
on the self-supervised task, you are getting better accuracy on the main task.
31:04
So the gains between those two tasks transfer and the self-supervised task doesn't need the label.
31:13
That's it. So this was one of the earliest cases of test-time compute
31:19
working in deep learning. And you might ask, this is interesting, Yu,
31:27
but it's only for object recognition. How could this idea still be relevant for hard problems
31:35
today in science and engineering? Well-- so let's think about how humans really
31:43
solve hard problems with only an unlabeled input.
31:48
And here's an experience you can all relate to, perhaps
31:54
think about your first really hard programming assignment. And this assignment is really hard
32:00
because you've gone to the lectures, done the homework, but it's just so much harder than what
32:06
you have seen over there. So how would you do it?
32:14
You at least need to try. You try, but you will stumble and fail.
32:20
But maybe you would-- the auto grader will show you a little sign of life, you've passed one test case or something.
32:26
You have a little partial success signal. And once you have all these search trajectories,
32:34
the important thing is you want to internalize this experience
32:41
so you can generalize a bit further in your new search trajectories.
32:46
And in AI terms, the first is just called search.
32:52
And the second is called learning. So when people do it in AI, we have
33:01
pre-training and post-training and prior work, even do search at test time by prompting
33:07
the LLM to attempt the problem and more sophisticated search
33:13
strategies. One of them is called AlphaEvolve. So the key idea of our work is to just train
33:22
on this new experience from a single problem. Now, this new experience is the most valuable experience
33:31
because it is specific to this problem. So it is exactly what you need to bridge the distribution
33:38
shift from over there to this particular direction. And let me give you an example to illustrate how it works.
33:47
So this is in GPU kernel engineering, say you are given a test problem in PyTorch
33:53
and that's what you tell the LLM, now write me a fast CUDA kernel implementation
33:59
of this piece of PyTorch code. And the reward in this case would then
34:04
be the runtime of-- the inverse reward would be the runtime of the kernel. You want it to run quickly.
34:10
And I will show you actual data recorded during our run, our test-time training run in a competition
34:18
called GPU mode. You can think of it as the Kaggle of kernel engineering.
34:24
And I will show the reward distribution over here and our goalpost is to exceed the best human.
34:32
So this is a result from prior work. So if you do search, say you sample 25,000 solutions from
34:41
this LLM with parameter theta 0, you get that reward distribution over there.
34:48
And this line over here shows what's the best solution if you just do search.
34:54
Now, our idea is to break this search budget into 50 rounds of search, each round with an improved model.
35:04
So we will still have the same total budget just, well, the first round we're only going to use 1/50 of the budget with
35:13
the initial model. So that's why you see the reward distribution is just a little bit squishy inside because we have a smaller
35:21
search. And you get theta 1, that's improved model,
35:27
using the search trajectories you just generated. And you do that by learning all the solutions reweighted
35:36
by their rewards. So this is going to reinforce the good solutions
35:43
and penalize the bad ones. Doing this will reinforce the good solutions
35:49
and penalize the bad ones. And this is a form of reinforcement learning just carried it out at test time.
35:55
AUDIENCE: I have a quick question. YU SUN: Yes, please. AUDIENCE: I thought the problem here was that at test time you don't have the solution, so you have
36:02
to learn from partial results. YU SUN: Yes. AUDIENCE: So what is the signal-- what is the reward signal in the partial results
36:08
that you're using. YU SUN: Yes. So in this particular case, a partial result
36:14
would be a kernel written by the LLM. It is still a valid kernel, but its runtime
36:21
would not be very good. It will be significantly worse than the best human
36:27
such as this runtime over here. That makes it a partial result. And the runtime itself
36:33
serves as the reward signal. AUDIENCE: And, in general, you would have to find something like this case by case?
36:40
YU SUN: In general, the LLM would be able to find this sort of a reward
36:46
when the problem definition naturally induces it. But there are, of course, problem definitions that do not
36:52
induce such kind of reward. Like if you're trying to prove P not equal to NP, for example,
36:58
you're not going to get that. [LAUGHS] So-- yes, please.
37:06
AUDIENCE: I assume that you would be able to do this type of test time training if the solution is verifiable?
37:13
YU SUN: If the solution is not only verifiable but also has continuous reward.
37:22
So P not equal to NP is it's very-- if you use a formal language verifier, you can verify it.
37:29
But the reward is pass or fail. Yeah. So I think the best intuition for where the algorithm would
37:37
apply is that any metric where you see humans
37:43
hill climbing on this is a natural thing to apply.
37:49
Yes, please. AUDIENCE: So you said continuous. Does it matter whether or it's actually smooth
37:55
or you have a gradient here, so-- YU SUN: The gradient does not matter. When I said continuous I just meant that there's
38:03
a wide range of such rewards. And this range is actually meaningful in the sense
38:08
that if you keep improving in this range-- I mean, it's distinct problems where humans can hill climb.
38:18
AUDIENCE: Sorry, I was just trying to get a sense because if there's 10 values that are ordered that's contiguous
38:24
from your perspective? YU SUN: Yeah, so if there's only 10 values, then you run into the possibility
38:31
that you might just get stuck at value 5, let's say.
38:37
If you search for a big budget at value 5
38:45
and everything still returns value 5, you're stuck. The key to making an improvement is that when you search,
38:55
there needs to be a spread of the values.
39:02
Cool. Yeah, thanks for the quest-- yes. AUDIENCE: Is there an analogous reward signal for the case
39:08
where you're just talking to a chatbot and you expect them to remember the previous context,
39:13
for example? Because it seems like that's the use
39:20
case of continual learning in-- YU SUN: I think it's a limited use case of continual learning.
39:26
And in that case, I think a reward signal is just how well you are at predicting the next token.
39:34
AUDIENCE: The human [INAUDIBLE]. YU SUN: The-- I mean, if you're trying to read, let's say,
39:40
a book that a human is giving you, then it's the same as trying to do what we call prefill.
39:46
So if you're trying to prefill this book, you're trying to read this book. Yeah, one natural reward signal as you read this book is how well you are
39:53
able to predict the next token. And this is indeed the reward signal
39:58
that we've been using for the previous project.
40:04
Just because this reward is fully differentiable as a process of predicting-- is fully differentiable,
40:10
we don't need to do reinforcement learning. Yeah, it's a nice connection that you drew?
40:17
Yeah, any other question? Cool, then let me go on.
40:25
So I was just talking about how you go from search, getting a range of trajectories with different reward values,
40:34
going from that to learning, that is just penalizing-- learning to-- reinforcing the good solutions
40:41
and penalizing the bad ones. And basically you just keep repeating those steps.
40:48
So if you do that-- and we inspect all the kernels. We have-- we inspect and we also have the competition organizers
40:54
inspect all the kernels that we have generated. So you see that theta 10, it learns
41:01
to consistently apply mixed precision to this particular problem. And you keep going, you see some other ideas
41:08
that VLM is able to implement and, finally, you will get a solution.
41:15
You will get a solution here that bets the best human. That's it.
41:20
Now, you see that the distributions are getting wider here.
41:25
So this is an important detail that is particular to test-time training.
41:31
So in classic RL, you want to produce a model that produces
41:39
good solutions on average. Namely you want to optimize for this quantity.
41:44
That's just the average reward of the actions drawn from your model. Now, this means that once you have a good solution-- in order
41:54
to make all solutions pretty good or the average solution is pretty good, you want to collapse to it.
42:01
And then the bad solutions before would become this good one and your solutions-- you have higher reward on average.
42:08
Now, our RL problem is somewhat special. We only care about one amazing solution to this problem.
42:19
In fact, we don't really care about the policy. It is nothing but a means to an end.
42:24
Our goal is just to get a solution. And, if I sample 500 solutions from the policy and 499 of them
42:34
are crap, I don't care about them, as long as there's one that's amazing.
42:39
So the way to say it in math is just if you sample 500,
42:45
all you care about is the performance of the best of them. That's it.
42:53
So we really have the extreme case of an exploration problem.
42:59
And exploration is a long-standing problem in RLs. There are many strategies to deal with it.
43:05
But in our case, because it is so important, I think the most principled way is just
43:11
to bake it into the objective function and optimize this thing literally.
43:17
Now, the max is hard to optimize. So we can just do a softmax instead with log sum
43:25
exponential. It's a standard trick in machine learning to implement a softer version of the max operator.
43:31
And that's what we do. So after we take care of this detail, our algorithm discovers a new best solution for every problem
43:41
that we have attempted in kernel engineering, mathematics, algorithm design, and biology.
43:46
And we had all the new solutions revealed by the organizers or domain experts.
43:53
So here-- yes, please. AUDIENCE: What did you use to generate the multiple solutions?
43:59
YU SUN: So you have an LLM and you prompt it
44:04
with is the original problem. And if you are building this on top of a slightly more
44:12
sophisticated search strategy, you would also prompt it with perhaps some solutions
44:18
that it has seen before. AUDIENCE: So you vary the prompts? YU SUN: Across each across each round, yes?
44:25
AUDIENCE: But do you do that in an automated fashion? How do you generate them? YU SUN: In an automated fashion. AUDIENCE: Yeah, but how?
44:31
YU SUN: Oh, so you are saying the prompt to-- so the prompt template is just taken from prior work.
44:40
So there are prior work that's doing this-- for example this Erdos problem I was just showing how they have a search
44:47
baseline, so we just take the same prompt template.
44:52
AUDIENCE: But, yeah-- sorry, but that's one template. Where do the variations come from? Are they random or--
44:58
YU SUN: They're random. Yeah, the variations are random. You fill in the template with a problem
45:04
and then the variations will come from some randomly sampled solutions from your past runs.
45:09
That's it. It is really simple actually, this way of doing it.
45:16
So this is a math problem--
45:22
is a problem in combinatorial number theory. It's called the Erdos minimum overlap problem.
45:29
And people are trying to improve this upper bound by making it lower. So lower is better.
45:34
And that was the previous human result
45:39
and DeepMind improved on this with their thing called AlphaEvolve and that was the best search baseline.
45:48
So a few months ago, Terry Tao collaborated with DeepMind and the latest version of AlphaEvolve
45:56
and improved it further. So if you just run an open model on this problem,
46:06
it's not going to do very well. But if you do test-time training,
46:12
it improves the result dramatically. It's a huge margin compared to what
46:18
they were able to do before. Yes, there are questions. Yes, please. AUDIENCE: Did we learn anything about the problem from the three
46:27
bars on the right of this? YU SUN: Yeah, so I did not learn anything,
46:35
but that was my problem, I would say. So we had math professors review our solution.
46:43
And I think that-- if you read the reviews in our paper--
46:48
so their center point was that actually all the solutions that the previous things have found are symmetric.
46:55
So to illustrate what that means, we're trying to find a step function that's
47:01
a constructive upper bound for these things they are doing. And basically the function they found are symmetric
47:08
and they say our thing is asymmetric. Hopefully they learn something, if that makes sense.
47:16
AUDIENCE: OK. YU SUN: I mean, it's hard for me to learn something from that.
47:23
Yes. AUDIENCE: For these math problems, did you try it with just one open model
47:30
or across a range of smaller models also and were the smaller--
47:35
YU SUN: We tried quite a lot of smaller models, yes. AUDIENCE: Were they able to improve upon it
47:40
or have a good solution? YU SUN: Yeah, I think this was-- I think we even had [INAUDIBLE].
47:48
And I'm not sure whether it was-- I need to check the paper whether it was improving upon this gray thing.
47:56
But it was definitely improving upon the green thing.
48:04
And if you ask whether TDT was improving the small model without TDT, yes, by a huge margin.
48:12
Yes. Yes, please. AUDIENCE: There a specific RL algorithms that you use?
48:20
Are you like-- is this policy optimization or-- YU SUN: Yeah, so a big difference
48:27
between the RL algorithms is just how they constrain the update.
48:34
Usually for PPO stuff, you do this KL thing kind of constraint, so you still keep that.
48:39
And then the most important is, how do you go from the raw rewards of this sort
48:51
to the actual weights of the solutions? So what I showed here, doing--
49:01
using this objective, would just give you a different way
49:07
to weigh the solutions. And that's called the entropic utility.
49:12
So you can think of us as doing a combination of RL techniques,
49:18
yes, so-- yeah. AUDIENCE: And how sample efficient was it? How many times did you have to prompt and get a reward on each?
49:26
YU SUN: Yeah, so really in this example each time index--
49:35
going from theta 1 to theta 2 means 1/50 of the search budget,
49:41
which is our total budget was 25,000, I believe.
49:47
And, yeah, it was 25,000. So 150 of that is 500. That's right, 500.
49:55
Yes, yes. AUDIENCE: So can we take questions later? We have five more minutes. YU SUN: OK, let me--
50:01
sorry about this. Now-- yeah, now I'm basically out of time, so let me wrap up.
50:12
This is what we talked about today. There's a moment when the test instance
50:19
arrives is really a defining moment in machine learning. And most machine learning in the past-- pre-training, and
50:27
post-training, mid-training, all these things-- they were about what to do before the test instance
50:36
arrives. And I think the future is about what to do after the test instance arrives because then you
50:41
no longer need to guess what the test instance is. And I talked about the simple setting
50:47
that's given the relevant data, also known as context. So I showed this pair of plots, which
50:52
was the first result, to the best of my knowledge, that scaling is with long context in both panels.
50:58
And what I didn't get to show are-- that we applied this technique to generate
51:05
one-minute long videos packed with a lot of dynamic motion.
51:10
And you can do this just by giving the model one, long plot
51:15
in text, a big storyboard. And it just generates the entire video in one shot with no human editing.
51:23
So this was the first model that was able to do that back in early 2025.
51:31
And long context is only the easiest form
51:36
of test-time training. Yeah, let me go on then from the video.
51:43
So now I also talked about the hard setting and there is no relevant data.
51:51
And in the hard setting, I talked about how to generate data through for self-supervision
51:57
and discovering state of the art results for scientific problems. What I did not get to talk about are
52:02
all the work I did in robotics, including manipulation, locomotion, and for perception.
52:10
And I will probably come back to robotics sometime in the future. Now, the most important thing is that I
52:16
didn't get to talk about the hundreds of direct follow-ups from other people since my paper in 2019.
52:23
And just to give you an example-- so AlphaProof was the first paper that
52:31
was able to win an IMO medal. And they say that for peak performance
52:37
in their most complex problems, test-time training is crucial. So they just prompt the LLM to generate some easier
52:45
exercises of the test problem that they actually have to solve.
52:51
And this technique is also applied in the leading math
52:57
theorem proving system today that's called Aristotle by Harmonic. And they also did a lot of experiments
53:05
on test-time training in their write-up.
53:10
So there have been many methods under the concept of test-time training. And most attribute my 2019 paper for bringing this concept
53:19
to deep learning. I think of this as my most important contribution.
53:25
So, in the future, I believe that the hard setting
53:31
with no relevant data is going to be the most powerful application of test-time training.
53:37
And, specifically, I like to solve problems. Start with verifiable problems with no continuous reward,
53:46
such as proving hard math theorems, because, I mean,
53:51
after all, that's what Andrew Wiles was able to do in those seven years. He was practically working in isolation,
53:59
just studying this problem. And can we formalize this process? So I'd like to start with verifiable problems
54:07
without dense reward, such as proving P not equal to NP.
54:13
And then this is actually only the easier among the hard problems.
54:19
The harder among the hard problems would to-- would involve interactions with the real world, such as robotics
54:27
and interacting with real users. Now, in the end, I believe that AI
54:34
can be the Andrew Wiles of the world's hardest and most important problems.
54:41
That's all my talk. Thank you. [APPLAUSE]
54:52
HANNA HAJISHIRZI: We have time for a couple more questions. Maybe we first start [INAUDIBLE].
54:58
AUDIENCE: Sure, yeah. That was a great talk. YU SUN: Thank you. AUDIENCE: [INAUDIBLE] HANNA HAJISHIRZI: No, they said no.
55:05
AUDIENCE: I guess the question was regarding the RL algorithm you were describing. I'm more familiar with the traditional RL,
55:12
but that has some clears of reward-weighted regression or that type of algorithm.
55:17
And I guess the question for-- have you--
55:25
how similar is it to that, but also-- is there much to learn from reward--
55:31
the RL methods used for more standard RL that can be applied to this as well? YU SUN: Yes.
55:39
So let me answer this question both in terms of what's being done now and in terms
55:45
of what will happen in the next maybe two to three years. So in terms of what's being done now, we--
55:55
here's how I try to do this research is I try to do single problem RL at test time.
56:04
That's my main idea. And for the rest of the research, I try to use as much of the default
56:10
RL techniques as possible so people would be able to focus on the main difference in isolation.
56:21
So, in this context, I mean, we basically try to borrow most of the traditional RL techniques.
56:28
Even the entropic utility, it has appeared in traditional RL. It just didn't matter that much in that context.
56:36
Over here, it matters a lot. Now, going on, I believe this sort of scenario
56:44
will become more important, this sort of borrowing.
56:50
Traditional RL thinks about sparse reward, sparse reward meaning you try the problem a lot of times,
56:57
you don't get much reward. That happens in traditional RL. Now, when you do test-time training,
57:03
it is a lot more extreme. When you have a binary reward problem, such as proving P not equal to NP, that is the most extreme.
57:11
And the techniques that people have developed in traditional RL for this sort of problem, one is called hindsight experience
57:19
replay. That's something I've been trying right now. Well, it's going to become more important in this scenario
57:27
and we might need to make it more extreme. So that's one of the relationship
57:34
I see both in terms of past project and in the future. AUDIENCE: Thank you. HANNA HAJISHIRZI: So one more question.
57:40
AUDIENCE: So everything you covered in this talk is there's one test example.
57:45
YU SUN: Yes. AUDIENCE: But in continual learning, you have an endless stream of examples, every one of which
57:50
is both training and test is in humans. YU SUN: Yes. AUDIENCE: How would you do things there, just
57:57
treat this one at a time? That seems like missing some opportunities. YU SUN: Yes, absolutely. So, in fact, while--
58:09
in a while-- while I was talking about the simple setting where you have long context, you are already
58:16
training the model online window by window in a long sequence.
58:23
And that will be a good idea in the future if you have a lot of test problems
58:30
that are highly related. And the biggest benefit for this is
58:37
that you will be able to save compute because the compute that you have spent for test-time training on earlier problem
58:44
could be used for a later problem if they are highly relevant to each other. HANNA HAJISHIRZI: So, in that case, isn't it like unsupervised learning
58:52
setting because you have a model trained, then it's like another stage of training
58:59
with just a group of examples without labels? YU SUN: Yeah. And the key idea is that each of these example
59:06
is actually a test instance. It's actually where you get evaluated.
59:12
It is-- the purpose of training on that test example is not only
59:18
to prepare for future ones. It's not to generalize to the other ones but to do well
59:25
in itself. Helping the future ones is only a byproduct of that.
59:33
HANNA HAJISHIRZI: OK. Thank you very much. Amazing talk. Thank you. YU SUN: Thank you everyone.
59:39
Thank you. [APPLAUSE]