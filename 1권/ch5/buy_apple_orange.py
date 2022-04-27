from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price =  mul_orange_layer.forward(orange, orange_num)
add_price = add_apple_orange_layer.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(add_price,tax)

# 역전파
price_d = 1
add_price_d, tax_d = mul_tax_layer.backward(price_d)
apple_price_d, orange_price_d  = add_apple_orange_layer.backward(add_price_d)
apple_d, apple_num_d = mul_apple_layer.backward(apple_price_d)
orange_d, orange_num_d = mul_orange_layer.backward(orange_price_d)

print("전체 가격 :", int(total_price))
print("사과 가격 미분:", apple_d)
print("사과 개수 미분:", int(apple_num_d))
print("오렌지 가격 미분:", orange_d)
print("오렌지 개수 미분:", int(orange_num_d))
print("소비세 미분:", tax_d)

