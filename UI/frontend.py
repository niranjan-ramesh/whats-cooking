import streamlit as st
import pandas as pd
from PIL import Image
from predict_similar_items import *
import circlify
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp


image = Image.open('./Data/img2.png')
usersdf = pd.read_csv("./Data/reccomendations_rbm.csv")
recipesdf = pd.read_csv("./Data/recipes_enriched.csv")
revdf = pd.read_csv("./Data/user_interactions.csv")
reviews = revdf
recipes = recipesdf

def bubblegraph(username):
    left =  reviews[reviews['user_id'] == username][['user_id', 'recipe_id']]
    right = recipes[['id', 'cuisine']]

    df = left.merge(right, left_on='recipe_id', right_on='id').groupby('cuisine', as_index=False).size().iloc[:10]
    print (df)

    color2=mcp.gen_color(cmap="hsv",n=len(df))
    circles = circlify.circlify(
        df['size'].tolist(), 
        show_enclosure=False, 
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    # Create just a figure and only one subplot
    fig, ax = plt.subplots(figsize=(10,10))

    # Title
    ax.set_title('Your Favourite cuisines!')

    # Remove axes
    ax.axis('off')

    # Find axis boundaries
    lim = max(
        max(
            abs(circle.x) + circle.r,
            abs(circle.y) + circle.r,
        )
        for circle in circles
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    # list of labels
    labels = df['cuisine']
    colors = color2
    cnt = 0

    # print circles
    for circle, label, color in zip(circles, labels, colors):
        cnt = cnt + 1
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2,facecolor=color, edgecolor="black"))
        plt.annotate(
              label, 
              (x,y ) ,
              va='center',
              ha='center'
         )
        if cnt == 10:
            st.pyplot(plt)

def item_to_item(recipesdf, modelval, page_select,username):
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
                  args=(recipesdf, rec_name, page_select, username))


def recipedetails(recipesdf, rec_name, page_select,username):
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
        item_to_item(recipesdf, recommendids, page_select,username)


def logincheck(usersdf, username, page_type):
    usersdf['user_id'] = usersdf['user_id'].apply(str)
    userval = usersdf[usersdf['user_id'].str.contains(str(username)) == True]
    modelval = list(userval.values[0, :10])
    if((userval.shape[0] >= 1)):
        bubblegraph(username)
        item_to_item(recipesdf, modelval, page_type,username)
    else:
        st.header("Oops! Wrong Username or Password")


def signupcheck(username, password, page_select):
    grouprev = revdf.groupby('recipe_id').max('rating').reset_index()
    sortedrev = grouprev.sort_values(ascending=False, by=['rating'])
    rev = sortedrev.head(100)
    rev1 = rev.drop('rating', axis=1)
    iddf = rev1.sample(n=10)
    modelval = iddf['recipe_id'].values.tolist()
    item_to_item(recipesdf, modelval, page_select,username)


def main():
    st.image(image, output_format='auto')
    page_select = st.radio('Login/Signup', ['Login', 'Signup'])
    st.markdown("""---""")

    if (page_select == 'Login'):
        st.header("Login")
        page_type = 'Login'
        userslist = usersdf['user_id'].tolist()
        username = st.selectbox('User Id', userslist)
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
