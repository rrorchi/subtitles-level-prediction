##
####################################################################
import func
import streamlit as st
import plotly.express as px
####################################################################

st.title('Определитель уровня языка субтитров')

st.markdown('Уровень понимания языка определяется по шкале CEFR, стоящей из шести уровней: \n- A1 - элементарный уровень понимания, \n- А2 - уровень ниже среднего, \n- В1 - средний, \n- В2 - выше среднего, \n- С1 - продвинутый \n- и С2 - полное понимание языка.')

st.markdown('Это приложение поможет определить уровень языка в фильме, от А2 до С1. Загрузите субтитры в формате `.srt` и обученная ML модель определит, какой уровень больше подходит для этого фильма. Также приложение покажет общее количество слов и количество слов каждого уровня в субтитре!')

uploaded_file = st.file_uploader("Загрузите .srt файл", accept_multiple_files=False, type='srt')

####################################################################
## чтение файла, обработка, предсказание, расчет статистик
if uploaded_file is not None:

	subtitle = func.open_sub(uploaded_file)
	subtitle = func.sub_preprocess(subtitle)
	
####################################################################
##
	prediction = func.get_prediction(subtitle)
	
	st.header('Уровень языка субтитров: ' + str(func.CATS[prediction[0][0]]))
	st.write('Уверенность в прогнозе уровня языка:')
	
	st.table(prediction[1].sort_values(by=' ', ascending=False).astype('str')+'%')
	
####################################################################
##	
	statistics = func.get_statistics(subtitle, func.CATS[prediction[0][0]])
	
	st.write('Вcего распознанных слов:', round(statistics['sub_words_total'][0]))
	st.write('Уникальных распознанных слов:', round(statistics['sub_words_uniq'][0]))
	
	fig = px.bar(statistics, x="dict_level", y="dict_words_uniq", height=400, width=600, barmode='group',
             title='Количество уникальных слов из словарей в загруженном субтитре',
			 text='dict_words_uniq',
             labels={'dict_words_uniq': ' ',
					 'dict_level': 'Уровень языка словаря'})
	st.plotly_chart(fig)
	
####################################################################
##
	statinfo = func.get_info(func.CATS[prediction[0][0]])
	
	fig2 = px.bar(statinfo, x="dict_level", y="dict_words_uniq", height=400, width=600, barmode='group',
				 title='Среднее количество уникальных слов из словарей в субтитрах уровня ' + func.CATS[prediction[0][0]],
				 text='dict_words_uniq',
				 labels={'dict_words_uniq': ' ', 
						 'dict_level': 'Уровень языка словаря'})
	st.plotly_chart(fig2)
##	
####################################################################