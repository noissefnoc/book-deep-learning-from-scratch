import unittest
import layer_native


class TestMulLayer(unittest.TestCase):
    """

    """
    def test_buy_apple(self):
        apple = 100
        apple_num = 2
        tax = 1.1

        # layer
        mul_apple_layer = layer_native.MulLayer()
        mul_tax_layer = layer_native.MulLayer()

        # forward
        apple_price = mul_apple_layer.forward(apple, apple_num)
        price = mul_tax_layer.forward(apple_price, tax)

        self.assertAlmostEqual(price, 220, delta=0.001)

        # backward
        dprice = 1
        dapple_price, dtax = mul_tax_layer.backward(dprice)
        dapple, dapple_num = mul_apple_layer.backward(dapple_price)

        self.assertAlmostEqual(dapple, 2.2, delta=0.001)
        self.assertAlmostEqual(dapple_num, 110, delta=0.001)
        self.assertAlmostEqual(dtax, 200, delta=0.001)

    def test_buy_apple_and_orange(self):
        apple = 100
        apple_num = 2
        orange = 150
        orange_num = 3
        tax = 1.1

        # layer
        mul_apple_layer = layer_native.MulLayer()
        mul_orange_layer = layer_native.MulLayer()
        add_apple_orange_layer = layer_native.AddLayer()
        mul_tax_layer = layer_native.AddLayer()

        # forward
        apple_price = mul_apple_layer.forward(apple, apple_num)
        orange_price = mul_orange_layer.forward(orange, orange_num)
        all_price = add_apple_orange_layer.forward(apple_price, orange_price)
        price = mul_tax_layer.forward(all_price, tax)

        self.assertAlmostEqual(apple_price, 200, delta=0.001)
        self.assertAlmostEqual(orange_price, 450, delta=0.001)
        self.assertAlmostEqual(all_price, 650, delta=0.001)
        self.assertAlmostEqual(price, 715, delta=0.001)

        # backward
        dprice = 1
        dall_price, dtax = mul_tax_layer.backward(dprice)
        dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
        dorange, dorange_num = mul_orange_layer.backward(dorange_price)
        dapple, dapple_num = mul_apple_layer.backward(dapple_price)

        self.assertAlmostEqual(dapple_num, 110, delta=0.001)
        self.assertAlmostEqual(dapple_price, 2.2, delta=0.001)
        self.assertAlmostEqual(dorange, 3.3, delta=0.001)
        self.assertAlmostEqual(dorange_num, 165, delta=0.001)