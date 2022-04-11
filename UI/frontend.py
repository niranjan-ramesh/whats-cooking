import streamlit as st
import pandas as pd
from PIL import Image
from predict_similar_items import *


image = Image.open('./Data/img2.png')
usersdf = pd.read_csv("./Data/user_pass.csv")
recipesdf = pd.read_csv("./Data/recipes_enriched.csv")
revdf = pd.read_csv("./Data/user_interactions.csv")


def item_to_item(recipesdf, modelval, page_select):
    file2 = open("session.txt", "w")
    file2.writelines("False")
    if (page_select == 'Signup'):
        st.header("Top Highly Rated Recipes!")

    if (page_select == 'Login'):
        st.header("Our Recommendations for you!")

    for i in range(len(modelval)):
        st.write("Recipe #", i+1)
        rec = recipesdf.loc[recipesdf['id'] == int(modelval[i])]
        rid = rec['id']
        print(rid)
        idval = [rid.values[0]]
        idval = list(idval)
        rname = rec['name']
        rec_name = rname.values[0]
        st.button(rec_name.title(), on_click=recipedetails,
                  args=(recipesdf, rec_name, page_select))


def recipedetails(recipesdf, rec_name, page_select):
    file2 = open("session.txt", "w")
    file2.writelines("False")
    rec = recipesdf.loc[recipesdf['name'] == rec_name]
    rid = rec['id']
    idval = [rid.values[0]]
    idval = list(idval)
    rname = rec['name']
    rec_name = rname.values[0]
    st.header(rec_name.title())
    mins = rec['minutes']
    prep = mins.values[0]
    st.subheader('Cooking Time in Minutes')
    st.text(prep)
    rec['ingredients'] = rec['ingredients'].apply(eval)
    val = rec['ingredients'].values[0]
    st.subheader('Ingredients')
    for j in range(len(val)):
        st.write("‣ ", val[j])
    rec['steps'] = rec['steps'].apply(eval)
    steps = rec['steps'].values[0]
    st.subheader('Steps to Cook')
    for k in range(len(steps)):
        st.write("Step ", k+1, "--->", steps[k])
    st.markdown("""---""")
    my_expander = st.expander("Similar Recipes", expanded=True)
    with my_expander:
        recommendids = get_similar_items(val)
        item_to_item(recipesdf, recommendids, page_select)


def logincheck(usersdf, username, page_type):
    usersdf['username'] = usersdf['username'].apply(str)
    userval = usersdf[usersdf['username'].str.contains(str(username)) == True]
    modelval = list(userval.values[0, 1:])
    if((userval.shape[0] >= 1)):
        item_to_item(recipesdf, modelval, page_type)
    else:
        st.header("Oops! Wrong Username or Password")


def signupcheck(username, password, page_select):
    grouprev = revdf.groupby('recipe_id').max('rating').reset_index()
    sortedrev = grouprev.sort_values(ascending=False, by=['rating'])
    rev = sortedrev.head(100)
    rev1 = rev.drop('rating', axis=1)
    iddf = rev1.sample(n=10)
    modelval = iddf['recipe_id'].values.tolist()
    item_to_item(recipesdf, modelval, page_select)


def main():
    file1 = open(r"session.txt", "r+")
    ses = file1.read().strip()
    print(ses)
    print(type(ses))
    if (ses == "True"):
        st.image(image, output_format='auto')
        page_select = st.radio('Login/Signup', ['Login', 'Signup'])
        st.markdown("""---""")

        if (page_select == 'Login'):
            st.header("Login")
            page_type = 'Login'
            userslist = usersdf['username'].tolist()
            username = st.selectbox('Username', userslist)
            password = st.text_input("Password", type='password')
            st.button('Login', on_click=logincheck,
                      args=(usersdf, username, page_type))

        if (page_select == 'Signup'):
            page_type = 'Signup'
            st.header("Sign up")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            st.button('Signup', on_click=signupcheck,
                      args=(username, password, page_type))


main()
