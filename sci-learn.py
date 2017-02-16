from sklearn.datasets import fetch_olivetti_faces
print "sci-learn"

class Product():
    def __init__(self, name,price):
        self.name = name
        self.price = price

def find_products():
    print "geting products"
    products = []

    product1 = Product("iphone", 8000)
    products.append(product1)

    product2 = Product("ipad", 5000)
    products.append(product2)

    product3 = Product ("ipod",3000)
    products.append(product3)

    product4 = Product
    product4.name = "MacBook Pro"
    product4.price = 20000
    products.append(product4)

    return products

for product in find_products():
    valid  = (product.name and product.price)
    assert (valid)
    print "product " + product.name + " costs: " + str(product.price) + " RMB"

# show that there are 4 products using python
number_of_products = 0
for product in find_products():
    number_of_products += 1
assert (number_of_products == 4)
if number_of_products == 4:
    print "True."
else:
    print "False."

print number_of_products


