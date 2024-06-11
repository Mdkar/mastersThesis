from faker import Faker
import pandas as pd
import random

# generate dataset of 1000 rows. Each row should have a unique id, a random name, a random email, a random phone number, and list of random products purchased (max 5 per order), and a random total price for the order.
fake = Faker()
Faker.seed(0)
random.seed(0)

nouns = [fake.word("noun") for _ in range(250)]
adjectives = [fake.word("adjective") for _ in range(250)]

data = []
for i in range(1000):
    cart = []
    noun_idx = random.randint(0, 249)
    adj_idx = random.randint(0, 249)
    # pick 5 products for the cart normally distributed around the noun and adjective indices
    for _ in range(5):
        noun_idx = int(random.normalvariate(noun_idx, 10) % 250)
        adj_idx = int(random.normalvariate(adj_idx, 10) % 250)
        cart.append(adjectives[adj_idx].capitalize() + " " + nouns[noun_idx].capitalize())

    data.append({
        'id': i,
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'credit_card': fake.credit_card_number(),
        'products': cart,
        'total_spent': fake.random_int(0, 1000)
    })

# save the data to a csv file
df = pd.DataFrame(data)
df.to_csv('train_data.csv', index=False)

# generate a test set of 250 rows
data = []
for i in range(250):
    cart = []
    noun_idx = random.randint(0, 249)
    adj_idx = random.randint(0, 249)
    # pick 5 products for the cart normally distributed around the noun and adjective indices
    for _ in range(5):
        noun_idx = int(random.normalvariate(noun_idx, 10) % 250)
        adj_idx = int(random.normalvariate(adj_idx, 10) % 250)
        cart.append(adjectives[adj_idx].capitalize() + " " + nouns[noun_idx].capitalize())

    data.append({
        'id': i,
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'credit_card': fake.credit_card_number(),
        'products': cart,
        'total_spent': fake.random_int(0, 1000)
    })
df = pd.DataFrame(data)
df.to_csv('test_data.csv', index=False)