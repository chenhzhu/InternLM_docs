import re

def wordcount(text: str):
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()

    word_count: dict[str, int] = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    return word_count



text1 = """Hello world!  
This is an example.  
Word count is fun.  
Is it fun to count words?  
Yes, it is fun!"""

result1 = wordcount(text1)
print(result1)

text2 = """
Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
"""

result2 = wordcount(text2)
print(result2)