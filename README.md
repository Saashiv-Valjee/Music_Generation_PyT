## Music Generation
----------------------------------

This was a ~1 day long attempt at using PyTorch to generate music. This uses the MAESTRO dataset. Previously I've attempted doing this using LSTMs and while using transformers definitely yields stronger results, I can't call it music yet. One key developement of the project was encoding the duration of the parsed notes alongside the note itself, while this did increase the vocab size it didn't have any major effects on the training duration or the results. \\

With more time, I'd have liked to do some research on the more state of the art techniques used to generate music. It is understandable that transformers can out perform LSTMs due to their attention mechanism, which happens to also be quite suited for music specifically. It would have been interesting to read on specific transformer architectures if I had the time. \\

As a side note, 
the code written here was mainly written via GPT. An evaluation of it's coding skills at the time of writing is that it can do very well at writing code IF! it's not got alot of conditions to remember. It's main drawback is it's memory so explaining it the entire project and telling it to write it all would never work. I believe that provided someone can understand and spot mistakes GPT is prone to, 
using it to increase productivity should be fine. An example of a mistake it made was forgetting that a function I had written had a condition to check for saved processed data. It defined a parameter in the function body whereas the parameter should only be defined within one of the conditions (if, else) which almost led to alot of wasted time. Fortunately having a working understanding of everything 
is (obviously) quite useful so I was able to solve this. \\

I am hoping I get the opportunity to develop music generation further at some point in the future. I'd like to be able to get it to a point where we have real-time music generation and have it applied to video games, so theme tracks can adjust depending on player and enemy interactions for example.
