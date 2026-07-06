
import requests  
import time 

data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
} 


start = time.time() 

response = requests.post("http://127.0.0.1:5000/predict",
                         json=data) 

end = time.time() 

print("Response:", response.json())  
print(f"time taken : {end - start :.4f} seconds")