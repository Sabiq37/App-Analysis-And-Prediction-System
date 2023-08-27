from flask import Flask, render_template, request,url_for,redirect
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

app = Flask(__name__)

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)
# Load the dataset
original_csv_file = "C:\\Users\\hp\\OneDrive\\Desktop\\ARS_Project\\GoogleApps.csv"
df_original = pd.read_csv(original_csv_file)
# Create a copy of the dataset
df = df_original.copy()

# Drop rows with missing values
df.dropna(inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Calculate correlation matrix
correlation_matrix = df.corr()
# Sort attributes by absolute correlation with the target variable ('Rating')
relevant_attributes = correlation_matrix['Rating'].abs().sort_values(ascending=False).index[1:]

# Assign the attributes with highest correlation to selected_features
selected_features = list(relevant_attributes)

# Assign the 'Rating' column to the target variable 'y'
y = df['Rating']

# Split the data
split_ratio = 0.8
split_index = int(len(df) * split_ratio)
X_train, X_test = df[selected_features].iloc[:split_index], df[selected_features].iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Apply PCA
pca = PCA(n_components=len(selected_features))
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Gaussian Kernel Locally Weighted Linear Regression
def gaussian_kernel(weights_bandwidth, x_query, x_data):
    weights = np.exp(-(np.linalg.norm(x_data - x_query, axis=1)**2) / (2 * weights_bandwidth**2))
    return weights

def gwlr_predict(x_query, X_train, y_train, weights_bandwidth):
    weights = gaussian_kernel(weights_bandwidth, x_query, X_train)
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict(x_query.reshape(1, -1))


# Load the unique categories
unique_categories = ['ART_AND_DESIGN', 'AUTOMOBILES', 'BEAUTY', 'BOOKS_AND_REFERENCE',
 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT',
 'EVENTS', 'FINANCE', 'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',
 'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL', 'SOCIAL',
 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL', 'TOOLS',
 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER', 'VIDEO_PLAYERS',
 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']

# Your authentication logic
admin_credentials = {"MD_SABIQ": "HELLO123"}
user_credentials = {"user": "1234"}

def authenticate_admin(username, password):
    if username in admin_credentials and admin_credentials[username] == password:
        return True
    return False

def authenticate_user(username, password):
    if username in user_credentials and user_credentials[username] == password:
        return True
    return False

# Routes
@app.route('/', methods=['GET', 'POST'])
def authenticate_web():
    if request.method == 'POST':
        user_type = request.form['user_type']
        username = request.form['username']
        password = request.form['password']
        
        if user_type == 'admin':
            if authenticate_admin(username, password):
                # Redirect to the admin menu
                return redirect('/admin_menu')
            else:
                message = "Authentication failed. Please try again."
        elif user_type == 'user':
            if authenticate_user(username, password):
                return redirect('/user_menu')
            else:
                message = "Authentication failed. Please try again."
        else:
            message = "Invalid input. Please select 'admin' or 'user'."

        return render_template('login.html', message=message)

    return render_template('login.html', message="")

@app.route('/admin_menu', methods=['GET', 'POST'])
def admin_menu():
    if request.method == 'POST':
        admin_choice = int(request.form['admin_choice'])
        
        # Inside the '/admin_menu' route, for admin choice 1
        if admin_choice == 1:
            return most_widely_used_app(X_test_pca, X_train_pca, y_train, df)

        elif admin_choice == 2:
            return redirect('/category_selection')
        elif admin_choice == 3:
            return redirect('/add_record')
        # Inside the '/admin_menu' route
        elif admin_choice == 4:
            return render_template('remove_record.html')

        elif admin_choice== 5:
            # Redirect back to the login page
            return redirect(url_for('authenticate_web'))

    return render_template('admin_menu.html')

@app.route('/user_menu', methods=['GET', 'POST'])
def user_menu():
    if request.method == 'POST':
        user_choice = int(request.form['user_choice'])
        
        if user_choice == 1:
            return most_widely_used_app(X_test_pca, X_train_pca, y_train, df)

        elif user_choice == 2:
            return redirect('/category_selection')
        
        elif user_choice == 3:
            # Redirect back to the login page
            return redirect(url_for('authenticate_web'))

    return render_template('user_menu.html') 


def most_widely_used_app(X_test_pca, X_train_pca, y_train, df):
    # Choose a query point for prediction
    query_point = X_test_pca[0]
    prediction = gwlr_predict(query_point, X_train_pca, y_train.values, weights_bandwidth=0.5)

            # Sort the dataframe by 'Installs' column in descending order
    most_widely_used_app = df.sort_values(by='Reviews', ascending=False).iloc[0]

    # Return the rendered HTML template
    return render_template('best_app.html', app=most_widely_used_app)


@app.route('/category_selection', methods=['GET', 'POST'])
def category_selection():
    if request.method == 'POST':
        user_choice = int(request.form['category'])
        if 1 <= user_choice <= len(unique_categories):
            selected_category = unique_categories[user_choice - 1]
            return redirect(url_for('best_app', category=selected_category))

    categories_with_index = [{'index': index + 1, 'category': category} for index, category in enumerate(unique_categories)]
    return render_template('category_selection.html', categories_with_index=categories_with_index)

@app.route('/best_app/<category>')
def best_app(category):
    selected_group = df[df['Category'] == category]
    best_app = selected_group.loc[selected_group['Reviews'].idxmax()]
    best_app_data = df[df['App'] == best_app['App']]
    return render_template('best_app_in_category.html', selected_category=category, best_app=best_app, best_app_data=best_app_data)

@app.route('/add_record', methods=['GET', 'POST'])
def add_record():
    if request.method == 'POST':
        new_record = {
            'App': request.form['app_name'],
            'Category': request.form['category'],
            'Rating': float(request.form['rating']),
            'Reviews': float(request.form['reviews']),
            'Size': float(request.form['size']),
            'Installs': float(request.form['installs']),
            'Type': request.form['type'],
            'Price': float(request.form['price']),
            'Content Rating': request.form['content_rating'],
            'Genres': request.form['genres'],
            'Last Updated': request.form['last_updated'],
            'Current Ver': float(request.form['current_ver']),
            'Android Ver': request.form['android_ver']
        }
        global df 
        df = df.append(new_record, ignore_index=True)
        df1=df.copy()
        df1.to_csv(original_csv_file, index=False)
        return render_template('record_added.html')
    return render_template('add_record.html')

def remove_record(app_name, category):
    global df_original,df
    
    if any((df_original['App'] == app_name) & (df_original['Category'] == category)):
        df = df[~((df['App'] == app_name) & (df['Category'] == category))]
        return True, f"Record for '{app_name}' in category '{category}' removed successfully."
    else:
        return False, f"No record found for '{app_name}' in category '{category}'."



@app.route('/remove_record', methods=['POST'])
def remove_record_route():
    app_name_to_remove = request.form['app_name']
    category_to_remove = request.form['category']
    record_removed, removal_message = remove_record(app_name_to_remove, category_to_remove)
    
    if record_removed:
        # Save the updated DataFrame back to the CSV file
        df.to_csv(original_csv_file, index=False)
        
    
    return render_template('record_removed.html', message=removal_message)


if __name__ == '__main__':
    app.run(debug=True)