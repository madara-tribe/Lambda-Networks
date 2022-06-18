# Lambda-Networks

<img width="887" alt="st" src="https://user-images.githubusercontent.com/48679574/174417435-1c095839-bc50-432e-99ab-dd995abfbdeb.png">

## Experiments
<img width="916" alt="スクリーンショット 2022-06-18 10 34 51" src="https://user-images.githubusercontent.com/48679574/174417440-a663ceba-00c2-40e0-8358-72c3c9f145ce.png">

<img width="813" alt="スクリーンショット 2022-06-18 10 35 03" src="https://user-images.githubusercontent.com/48679574/174417444-ebc6626f-54f9-4cbe-bb83-0c82a488766f.png">

# Versions
- Python 3.7.0
- tensorflow 2.3.0

# Performance

<b>task ：　Classify 11 color type and 2 shape type from 1 image</b>

| Layers | param | accuracy(color's classify) |
| :---         |     :---:      |          ---: |
| without LambdaNetwork   | 4,460,013     | 65.02%    |
| LambdaNetwork (+2).     | 4,660,461       | 72.31%(+7.29)    |
| LambdaNetwork (+4).     | 4,692,621       | nnn     |


## Model without lambda-network 

<b>Loss / Accuracy </b>

color, shape acc 0.6502636203866432 0.827768014059754

<img src="https://user-images.githubusercontent.com/48679574/174418438-6739db05-685d-4a34-b715-aeab1c81df7d.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/174418441-960ee8b4-5819-493a-95e4-57d38a149f61.png" width="400px">

## Model with lambda-network 

<img src="https://user-images.githubusercontent.com/48679574/174418583-98d0da06-6978-426c-88b7-657bda65394e.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/174418585-93933c75-8bdc-432f-972c-cb6db154704b.png" width="400px">


# References
-[LAMBDANETWORKS: MODELING LONG-RANGE INTERACTIONS WITHOUT ATTENTION](https://arxiv.org/pdf/2102.08602.pdf)


