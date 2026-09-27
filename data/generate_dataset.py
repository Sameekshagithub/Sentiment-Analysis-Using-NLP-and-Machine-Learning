"""
generate_dataset.py

Builds a synthetic dataset of product/service reviews labeled as
positive, neutral, or negative. There is no real customer data behind
this - it's assembled from templates and word lists so the rest of the
project has something realistic to train on without needing to scrape
or download anything.

Run this once to (re)create data/reviews_dataset.csv.
"""

import csv
import random
import os

random.seed(42)

SUBJECTS = [
    "product", "phone", "laptop", "headphones", "service", "app",
    "delivery", "food", "movie", "book", "course", "hotel", "camera",
    "software", "customer support", "packaging", "restaurant", "game",
    "watch", "speaker", "software update", "subscription", "website",
    "shoes", "jacket", "chair", "monitor", "keyboard", "mattress",
    "vacuum cleaner", "coffee machine", "trip", "flight", "warranty",
]

POSITIVE_ADJECTIVES = [
    "excellent", "amazing", "fantastic", "wonderful", "great", "superb",
    "impressive", "outstanding", "brilliant", "reliable", "smooth",
    "comfortable", "fast", "well made", "user friendly", "worth every penny",
    "beyond my expectations", "top notch", "solid", "delightful",
]

NEGATIVE_ADJECTIVES = [
    "terrible", "disappointing", "awful", "poor", "frustrating", "slow",
    "unreliable", "cheaply made", "confusing", "overpriced", "broken",
    "uncomfortable", "buggy", "a waste of money", "annoying", "useless",
    "not worth it", "second rate", "flimsy", "underwhelming",
]

NEUTRAL_ADJECTIVES = [
    "okay", "average", "decent", "acceptable", "fine", "so so",
    "nothing special", "reasonable", "standard", "as expected",
    "not bad but not great", "fair", "ordinary", "middle of the road",
]

POSITIVE_TEMPLATES = [
    "The {subject} is {adj} and I love it.",
    "I am really happy with this {subject}, it was {adj}.",
    "Absolutely {adj} {subject}, would buy again.",
    "This {subject} exceeded my expectations, {adj} experience overall.",
    "I really enjoyed the {subject}, it felt {adj}.",
    "Great {subject}! Everything about it was {adj}.",
    "Highly recommend this {subject}, truly {adj}.",
    "The {subject} works perfectly and feels {adj}.",
    "I was pleasantly surprised by how {adj} the {subject} turned out to be.",
    "Five stars for this {subject}, it is {adj}.",
    "The {subject} arrived early and looked {adj}.",
    "So glad I chose this {subject}, it is {adj} in every way.",
]

NEGATIVE_TEMPLATES = [
    "The {subject} is {adj} and I regret buying it.",
    "I am not happy with this {subject}, it was {adj}.",
    "Completely {adj} {subject}, would not recommend.",
    "This {subject} did not meet my expectations, {adj} experience overall.",
    "I did not enjoy the {subject} at all, it felt {adj}.",
    "Bad {subject}! Everything about it was {adj}.",
    "I would not recommend this {subject}, it was {adj}.",
    "The {subject} barely works and feels {adj}.",
    "I was disappointed by how {adj} the {subject} turned out to be.",
    "One star for this {subject}, it is {adj}.",
    "The {subject} arrived late and looked {adj}.",
    "I regret choosing this {subject}, it is {adj} in every way.",
    "I do not like this {subject}, it is {adj}.",
    "This is not a good {subject}, the quality is {adj}.",
]

NEUTRAL_TEMPLATES = [
    "The {subject} is {adj}, nothing more to say.",
    "I have mixed feelings about this {subject}, it was {adj}.",
    "The {subject} was {adj}, it does the job.",
    "This {subject} is {adj}, neither good nor bad.",
    "It is a {adj} {subject}, might consider other options next time.",
    "The {subject} is {adj} for the price.",
    "My experience with the {subject} was {adj}.",
    "The {subject} performed in an {adj} way, no strong opinion either way.",
    "It's a {adj} {subject}, similar to others I have tried.",
    "The {subject} is {adj}, I don't have strong feelings about it.",
]

# A handful of hand written, less templated lines per class so the model
# does not only learn the exact template shapes above.
EXTRA_POSITIVE = [
    "Honestly did not expect much but this blew me away.",
    "Customer support solved my issue in five minutes, could not ask for more.",
    "Battery lasts two full days now, huge improvement over the old one.",
    "My kids won't stop playing with it, best purchase this year.",
    "Setup took less than ten minutes and it just works.",
    "The build quality feels premium for the price point.",
    "Never had a single issue after three months of daily use.",
    "The staff went out of their way to make sure we were comfortable.",
    "Shipping was quick and the item was packed with a lot of care.",
    "This completely changed how I get through my morning commute.",
]

EXTRA_NEGATIVE = [
    "Stopped working after just two days, total waste of money.",
    "Customer support kept transferring my call and never solved anything.",
    "Battery drains within a couple of hours even with light use.",
    "My kids lost interest within a day, would not buy again.",
    "Setup took over an hour and the instructions made no sense.",
    "The build quality feels cheap for what I paid.",
    "Had three separate issues within the first month of use.",
    "The staff was rude and did not seem to care about our concerns.",
    "Shipping took forever and the box arrived crushed.",
    "This made my daily commute more stressful, not less.",
]

EXTRA_NEUTRAL = [
    "It works as described, nothing to complain about but nothing exciting either.",
    "Customer support answered eventually, took a couple of days.",
    "Battery life is about what I expected, nothing more.",
    "My kids play with it sometimes, other toys get more attention.",
    "Setup was straightforward, took about twenty minutes.",
    "The build quality is fine for casual use.",
    "Had one small issue in three months, got resolved.",
    "The staff was polite, service was average overall.",
    "Shipping arrived on the estimated date, no surprises.",
    "It fits into my routine fine, does not stand out much.",
]

NEGATIONS_POSITIVE_TO_NEGATIVE = [
    "I do not think this {subject} is {adj} at all.",
    "It is hard to call this {subject} {adj}, honestly.",
    "This {subject} is not as {adj} as the reviews suggested.",
]


def build_rows():
    rows = []

    for subject in SUBJECTS:
        for template in POSITIVE_TEMPLATES:
            adj = random.choice(POSITIVE_ADJECTIVES)
            rows.append((template.format(subject=subject, adj=adj), "Positive"))
        for template in NEGATIVE_TEMPLATES:
            adj = random.choice(NEGATIVE_ADJECTIVES)
            rows.append((template.format(subject=subject, adj=adj), "Negative"))
        for template in NEUTRAL_TEMPLATES:
            adj = random.choice(NEUTRAL_ADJECTIVES)
            rows.append((template.format(subject=subject, adj=adj), "Neutral"))

    # a few negated positive adjectives that actually read negative,
    # so the model sees some examples where "not" flips the meaning
    for subject in random.sample(SUBJECTS, 12):
        template = random.choice(NEGATIONS_POSITIVE_TO_NEGATIVE)
        adj = random.choice(POSITIVE_ADJECTIVES)
        rows.append((template.format(subject=subject, adj=adj), "Negative"))

    for text in EXTRA_POSITIVE:
        rows.append((text, "Positive"))
    for text in EXTRA_NEGATIVE:
        rows.append((text, "Negative"))
    for text in EXTRA_NEUTRAL:
        rows.append((text, "Neutral"))

    random.shuffle(rows)
    return rows


def main():
    rows = build_rows()
    out_path = os.path.join(os.path.dirname(__file__), "reviews_dataset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    counts = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    print("Class distribution:", counts)


if __name__ == "__main__":
    main()
