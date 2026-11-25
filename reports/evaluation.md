# Evaluation - `word_1-2gram`

## Held-out test set

```
              precision    recall  f1-score   support

      cement       1.00      1.00      1.00         3
       steel       0.50      0.67      0.57         3
  aggregates       1.00      0.67      0.80         3
      blocks       0.67      0.50      0.57         4
    services       0.40      0.50      0.44         4

    accuracy                           0.65        17
   macro avg       0.71      0.67      0.68        17
weighted avg       0.69      0.65      0.66        17

```
```
                 cement  steel  aggregates  blocks  services
true:cement           3      0           0       0         0
true:steel            0      2           0       0         1
true:aggregates       0      1           2       0         0
true:blocks           0      0           0       2         2
true:services         0      1           0       1         2
```

## 5-fold out-of-fold (all lines)

```
              precision    recall  f1-score   support

      cement       0.92      0.86      0.89        14
       steel       0.71      0.77      0.74        13
  aggregates       0.73      0.67      0.70        12
      blocks       0.83      0.71      0.77        14
    services       0.61      0.73      0.67        15

    accuracy                           0.75        68
   macro avg       0.76      0.75      0.75        68
weighted avg       0.76      0.75      0.75        68

```
```
                 cement  steel  aggregates  blocks  services
true:cement          12      0           0       0         2
true:steel            0     10           1       0         2
true:aggregates       0      2           8       1         1
true:blocks           0      0           2      10         2
true:services         1      2           0       1        11
```

## Mistakes (17) - most confident first

| text | true | predicted | confidence |
|---|---|---|---|
| Steel shuttering plate rental | services | steel | 0.79 |
| خرسانة جاهزة متر مكعب | cement | services | 0.49 |
| Kerb stone 50cm | blocks | aggregates | 0.47 |
| Rebar cutting and bending service | services | steel | 0.46 |
| حصى مغسول ٢٠ مم | aggregates | steel | 0.44 |
| طوب احمر | blocks | aggregates | 0.42 |
| ايجار مضخة خرسانة | services | cement | 0.40 |
| Ready-mix concrete C40 per m3 | cement | services | 0.38 |
| بحص مكسر ١٠ مم | aggregates | steel | 0.38 |
| Black annealed binding wire | steel | aggregates | 0.37 |
| Crushed stone 10mm | aggregates | blocks | 0.33 |
| Pump hire for concrete pouring | services | blocks | 0.29 |
| Road base sub-base material | aggregates | services | 0.24 |
| حجر رصيف | blocks | services | 0.24 |
| سلك ربط مجلفن | steel | services | 0.24 |
| GI pipe 2 inch 6m length | steel | services | 0.23 |
| Red clay brick | blocks | services | 0.23 |