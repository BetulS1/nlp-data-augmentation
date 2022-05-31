#python library
from googletrans import Translator
import numpy as np, pandas as pd
from dask import bag, diagnostics

if __name__ == '__main__':

#*************** cümleyi dillere göre çevirir ardından ilk tespit edilen dile göre listeler **********
    def Translate_Google(sequence, PROB = 1):
        languages = ['en', 'fr', 'tr', 'ru', 'bg', 'de', 'es', 'el']
        translator = Translator()
        org_lang = translator.detect(sequence).lang
        random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
        translated = translator.translate(sequence, dest = random_lang).text
        translated_back = translator.translate(translated, dest = "tr" ).text
        if np.random.uniform(0, 1) <= PROB:
            output_sequence = translated_back
        else:
            output_sequence = sequence

        return output_sequence

    def CSV_Translate_Google(dataset):
        
        narrative_list =  bag.from_sequence(dataset['narrative'].tolist()).map(Translate_Google)
        product_list =  bag.from_sequence(dataset['product'].tolist())
         
        with diagnostics.ProgressBar():
            translate_narrative = narrative_list.compute()
            product = product_list.compute()

        print("İşleme Başlatıldı Lütfen Bekleyiniz!....")

        row_narrative = translate_narrative
        row_product = product
        new_data=[]
        new_product=[]
        for x in range(5) :
            for i in range(len(row_narrative)):
                    output = Translate_Google(row_narrative[i])
                    new_data.append(output)       
                    new_product.append(row_product[i])       
                    
        veriler = {"product": new_product, "narrative": new_data}
        dataset = pd.DataFrame(veriler)
        return dataset

    data_csv = pd.read_csv('complaints.csv')
    yeni_data = data_csv.pipe(CSV_Translate_Google)
    yeni_data.to_csv('Translate_complaints.csv')
    print("İşleme Tamamlandı!....")
    print("Dosya Oluşturuldu!....")
    
 
