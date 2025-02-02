import pandas as pd

food_dict = {
    "Apple Pie": ['Apple', 'Milk', 'Other items',None],
    "Pizza": ['Olives', 'Cheddar Cheese', 'Mozzarella Cheese', 'Boneless Chicken'],
    "Banana Shake": ['Banana', 'Milk', None, None]  # Adding None to match column length
}

df = pd.DataFrame(food_dict)
print(df)
print(df.loc['Pizza'])
