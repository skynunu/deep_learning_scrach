from layer_naive import * 

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)


#backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("전체 가격 :", int(price))
print("사과 가격 미분:", dapple)
print("사과 개수 미분:", int(dapple_num))
print("소비세 미분:", dtax)