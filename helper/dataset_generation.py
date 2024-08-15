import random
import pandas as pd

positive_templates = [
    "I loved the {food_item}, and the delivery was {delivery_condition}.",
    "The {food_item} was {adjective}, and it arrived {delivery_condition}.",
    "Amazing {food_item}, delivered {delivery_condition}!",
]

neutral_templates = [
    "The {food_item} was okay, and the delivery was {delivery_condition}.",
    "The {food_item} was {adjective}, but the delivery was {delivery_condition}.",
    "The {food_item} was {adjective}. The delivery could have been better.",
]

negative_templates = [
    "I hated the {food_item}, and the delivery was {delivery_condition}.",
    "The {food_item} was {adjective}, and the delivery was {delivery_condition}.",
    "Terrible {food_item}, delivered {delivery_condition}.",
]

food_items = ["pizza", "sushi", "burger", "pasta", "salad"]
positive_adjectives = ["delicious", "perfect", "amazing", "tasty"]
neutral_adjectives = ["okay", "average", "decent", "acceptable"]
negative_adjectives = ["bland", "overcooked", "cold", "soggy"]
positive_delivery = ["fast", "on time", "early"]
neutral_delivery = ["a bit late", "acceptable", "slightly delayed"]
negative_delivery = ["late", "cold", "delayed"]


def assign_food_ratings(sentiment):
    if sentiment in positive_adjectives:
        food_rating = random.randint(4, 5)
    elif sentiment in neutral_adjectives:
        food_rating = random.randint(3, 4)
    elif sentiment in negative_adjectives:
        food_rating = random.randint(1, 2)
    return food_rating


def assign_delivery_ratings(sentiment):
    if sentiment in positive_delivery:
        delivery_rating = random.randint(4, 5)
    elif sentiment in neutral_delivery:
        delivery_rating = random.randint(3, 4)
    elif sentiment in negative_delivery:
        delivery_rating = random.randint(1, 2)
    return delivery_rating


def assign_approval():
    return random.randint(0, 1)


def generate_review(templates, adjectives, delivery_conditions):
    template = random.choice(templates)
    food_item = random.choice(food_items)
    adjective = random.choice(adjectives)
    delivery_condition = random.choice(delivery_conditions)
    food_rating = assign_food_ratings(adjective)
    delivery_rating = assign_delivery_ratings(delivery_condition)
    approval=assign_approval()
    return template.format(
        food_item=food_item,
        adjective=adjective,
        delivery_condition=delivery_condition,
    ), food_rating, delivery_rating,approval


positive_reviews = [generate_review(positive_templates, positive_adjectives, positive_delivery) for _ in range(50)]
neutral_reviews = [generate_review(neutral_templates, neutral_adjectives, neutral_delivery) for _ in range(20)]
negative_reviews = [generate_review(negative_templates, negative_adjectives, negative_delivery) for _ in range(50)]

positive_data = [{"review": review, "food_rating": food, "delivery_rating": delivery, "approval":approval}
                 for review, food, delivery, approval in positive_reviews]

neutral_data = [{"review": review, "food_rating": food, "delivery_rating": delivery,"approval":approval}
                for review, food, delivery, approval in neutral_reviews]

negative_data = [{"review": review, "food_rating": food, "delivery_rating": delivery,"approval":approval}
                 for review, food, delivery, approval in negative_reviews]

all_data = positive_data + neutral_data + negative_data
random.shuffle(all_data)
df = pd.DataFrame(all_data)
df.to_csv("generated_data.csv")
