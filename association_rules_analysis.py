#!/usr/bin/env python
# coding: utf-8

# # Python ile Birliktelik Kuralları Analizi

# In[71]:


# Gerekli kütüphane ve modüllerin yüklenmesi
import pandas as pd
import numpy as np
import mlxtend as ml
import random

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ## Veriseti Oluşturma

# In[72]:


dataset = []

# Elimizde 40 adet ürün kodu var
urunler = [ 'K1U0001',
            'K1U0002',
            'K1U0003',
            'K1U0004',
            'K1U0005',
            'K1U0006',
            'K1U0007',
            'K1U0008',
            'K1U0009',
            'K1U0010',
            'K1U0011',
            'K1U0012',
            'K1U0013',
            'K1U0014',
            'K1U0015',
            'K1U0016',
            'K1U0017',
            'K1U0018',
            'K1U0019',
            'K1U0020',
            'K2U0001',
            'K2U0002',
            'K2U0003',
            'K2U0004',
            'K2U0005',
            'K2U0006',
            'K2U0007',
            'K2U0008',
            'K2U0009',
            'K2U0010',
            'K2U0011',
            'K2U0012',
            'K2U0013',
            'K2U0014',
            'K2U0015',
            'K2U0016',
            'K2U0017',
            'K2U0018',
            'K2U0019',
            'K2U0020']

# Elimizdeki ürün kodlarıyla 1000 tane rastgele gözlem kümesi oluşturuyoruz
for i in range(1, 1001):
    k = random.randrange(1, len(urunler)+1)
    dataset.append(random.choices(urunler, k=k))
    
    


# In[73]:


# Oluşturulan gözlem kümelerini dataset.csv dosyasına tablo şeklinde aktarıyoruz
df = pd.DataFrame(dataset)
df.to_csv('dataset.csv', index=False)


# In[74]:


# Gözlem sayısımızı kontrol ediyoruz
print(len(df))


# In[75]:


# Oluşturduğumuz tablodan ilk 5 gözlem değerini göstermek için
df.head()


# In[76]:


# Oluşturduğumuz tablodan son 5 gözlem değerini göstermek için
df.tail()


# In[77]:


# Veri setini oluşturduktan sonra nested list (iç içe liste) tipindeki verilerimizi tablosal bir yapıya çevirmemiz gerekiyor
# Bunun için mlxtend modülü içerisinde yer alan preprocessing sınıfı içerisinde TransactionEncoder fonksiyonunu kullanacağız

t = TransactionEncoder()
td = t.fit(dataset).transform(dataset)
df = pd.DataFrame(td, columns=t.columns_)


# In[78]:


# Tabloya çevirdiğimiz verisetimizin ilk beş değerini tekrar kontrol ediyoruz
df.head()


# ## Model Oluşturma

# Modeli oluşturmak için APRIORI algoritmasını kullanacağız

# In[79]:


# En az %20 destek şartımızla algoritmayı çalıştırıyoruz
apriori(df, min_support=0.2)


# In[80]:


# Yukarıdaki tabloda "itemset" kolunundaki numaralar yerin ürün kodlarını göstermek için "use_colnames=True" kod parçasını ekliyoruz
apriori(df, min_support=0.20, use_colnames=True)


# In[81]:


# Kaç tane gözlemin belirlediğimiz %20lik destek kuralımıza uyguduğunu kontrol etmek için 
print(len(apriori(df, min_support=0.20)))


# In[82]:


# Gözlem değerlerini atama 
gozlem_kumesi = apriori(df, min_support=0.20, use_colnames=True)

# Gözlem kümemizde her bir gözlemin kaç değer içerdiğini içeren bir kolon eklemek için
gozlem_kumesi['length'] = gozlem_kumesi['itemsets'].apply(lambda x: len(x))
gozlem_kumesi


# In[83]:


# Şimdi istediğimiz kriterlere uyan gözlemi daha rahat bir şekilde bulabiliriz
# Örnek: İçinde 2 ürün olan ve destek değeri %20 olan gözlem kümesi için

gozlem_kumesi[ (gozlem_kumesi['length'] == 2) & (gozlem_kumesi['support'] >= 0.2) ]


# In[85]:


# Veya ürünler bazında filtreleme yapabiliriz
gozlem_kumesi[ gozlem_kumesi['itemsets'] == {'K1U0001', 'K1U0007'} ]


# #### İlgilendiğimiz metriğe göre (confidence, lift, conviction ve vd.) Association Rules tablosunu oluşturmak için

# In[86]:


# Burada metrik olarak Confidence ve değerini 0.3 (%30) seçtik.
rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.30)


# In[87]:


print("Oluşan Kural Sayısı:", len(rules1))


# In[88]:


# Confidence değeri alçalan şekilde tabloda sıralamak için
rules1 = rules1.sort_values(['confidence'], ascending=False)
rules1


# #### Yorum: ID bilgisi 66 olan satırı inceleyecek olursak;
# 
K1U0001 ve K2U0019 ürünlerinin birlikte görülme olasılığı (support) %19 (0.19) olduğunu,
K1U0001 ürününü satın alan kişilerin yaklaşık (confidence) %54’ünün (0.539106) olasılıkla K2U0019 ürününü de satın aldığını,
K1U0001 ürününün yer aldığı alışveriş sepetlerinde K2U0019 ürününün satışı (lift) 1.316790 kat arttığı,
K1U0001 ve K2U0019 ürünlerinin birlikte satın alınmalarının ile birbirlerinden bağımsız olarak satın alınmalarından ne kadar fazla (leverage) 0.046478 olduğunu,
K1U0001 ve K2U0019 ürünlerinin birbirleri ile ilişkili olduğunu (conviction) 1.281403 değeri ile söyleyebiliriz.
# In[ ]:




