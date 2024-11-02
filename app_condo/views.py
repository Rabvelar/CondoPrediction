import pandas as pd
import xgboost as xgb
from django.shortcuts import render,get_object_or_404
from django.http import JsonResponse
from app_condo.models import District, Subdistrict, NearestRoad,CondoPricePrediction
import logging
import pickle
from xgboost import XGBRegressor
import os
from django.conf import settings


logger = logging.getLogger(__name__)

# Load the model
model = XGBRegressor()
model.load_model(os.path.join(settings.BASE_DIR, 'app_condo', 'data', 'xgb_model.json'))

# Load the label encoders
le_file_path = os.path.join(settings.BASE_DIR, 'app_condo', 'data', 'label_encoders.pkl')

with open(le_file_path, 'rb') as le_file:
    label_encoders = pickle.load(le_file)
def get_facilities():
    return [
        'Swimming Pool', 'Car Park', 'CCTV', 'Fitness',
        'Library', 'Security', 'Mini Mart', 'Electrical Sub Station'
    ]

def prediction_form(request):
    districts = District.objects.all().order_by('name')
    distance_fields = {
        'train_station': 0,
        'airport': 0,
        'university': 0,
        'department_store': 0,
        'hospital': 0
    }
    context = {
        'districts': districts,
        'distance_fields': distance_fields,
        'facilities': get_facilities(),
    }
    return render(request, 'predict.html', context)

def get_subdistricts(request, district_id):
    """Get subdistricts for the selected district."""
    try:
        subdistricts = Subdistrict.objects.filter(district_id=district_id).order_by('name')
        data = [{'id': s.id, 'name': s.name} for s in subdistricts]
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error getting subdistricts: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)

def get_nearest_roads(request, subdistrict_id):
    """Get nearest roads for the selected subdistrict."""
    try:
        roads = NearestRoad.objects.filter(subdistrict_id=subdistrict_id).order_by('name')
        data = [{'id': r.id, 'name': r.name} for r in roads]
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error getting nearest roads: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)
    
def safe_transform(label_encoder, value):
    """Transform a value with a label encoder, handling unseen labels."""
    try:
        return label_encoder.transform([value])[0]
    except:
        logger.warning(f"Unknown category {value}, using default encoding")
        return label_encoder.transform([label_encoder.classes_[0]])[0]

def predict(request):
    if request.method == 'POST':
        try:
            data = request.POST.dict()
            
            # Get the selected district
            district_name = data.get('district')
            district = District.objects.get(name=district_name)
            
            # Get subdistricts for the selected district
            subdistricts = Subdistrict.objects.filter(district=district)
            
            # Get nearest roads for the selected subdistrict
            subdistrict = Subdistrict.objects.get(name=data.get('subdistrict'), district=district)
            nearest_roads = NearestRoad.objects.filter(subdistrict=subdistrict)

            room_sizes = {
                "Studio": 24,
                "1 Bedroom": 35,
                "2 Bedrooms": 60,
                "3 Bedrooms": 100
            }

            # Prepare features dictionary
            features = {
                'District': data['district'],
                'Subdistrict': data['subdistrict'],
                'NearestRoad': data['nearest_road'],
                'BuildingAge': float(data['building_age']),
                'TotalUnits': float(data['total_units']),
                'TrainStation': float(data.get('train_station', 0)),
                'Airport': float(data.get('airport', 0)),
                'University': float(data.get('university', 0)),
                'Departmentstore': float(data.get('department_store', 0)),
                'Hospital': float(data.get('hospital', 0)),
                'Swimmingpool': 1 if data.get('swimming_pool') else 0,
                'CarPark': 1 if data.get('car_park') else 0,
                'CCTV': 1 if data.get('cctv') else 0,
                'Fitness': 1 if data.get('fitness') else 0,
                'Library': 1 if data.get('library') else 0,
                'Security': 1 if data.get('security') else 0,
                'MiniMart': 1 if data.get('mini_mart') else 0,
                'ElectricalSubStation': 1 if data.get('electrical_sub_station') else 0
            }

            
            features['District'] = safe_transform(label_encoders['District'], data['district'])
            features['Subdistrict'] = safe_transform(label_encoders['Subdistrict'], data['subdistrict'])
            features['NearestRoad'] = safe_transform(label_encoders['NearestRoad'], data['nearest_road'])

            # Create input DataFrame with correct column order
            input_df = pd.DataFrame([features], columns=[
                'NearestRoad', 'TrainStation', 'University', 'Airport', 
                'Departmentstore', 'Hospital', 'Subdistrict', 'District', 
                'TotalUnits', 'BuildingAge', 'CarPark', 'CCTV', 
                'Fitness', 'Library', 'Swimmingpool', 'Security', 
                'MiniMart', 'ElectricalSubStation'
            ])

            # Make prediction
            predicted_psm = float(model.predict(input_df)[0])
            room_size = room_sizes[data['room_size']]
            total_price = predicted_psm * room_size

            # Prepare context with all necessary data
            context = {
                'districts': District.objects.all().order_by('name'),
                'subdistricts': subdistricts,
                'nearest_roads': nearest_roads,
                'selected_district': district_name,
                'selected_subdistrict': data['subdistrict'],
                'selected_nearest_road': data['nearest_road'],
                'distance_fields': {
                    'train_station': data.get('train_station', 0),
                    'airport': data.get('airport', 0),
                    'university': data.get('university', 0),
                    'department_store': data.get('department_store', 0),
                    'hospital': data.get('hospital', 0)
                },
                'facilities': get_facilities(),
                'facility_values': {
                    'Swimming Pool': bool(data.get('swimming_pool')),
                    'Car Park': bool(data.get('car_park')),
                    'CCTV': bool(data.get('cctv')),
                    'Fitness': bool(data.get('fitness')),
                    'Library': bool(data.get('library')),
                    'Security': bool(data.get('security')),
                    'Mini Mart': bool(data.get('mini_mart')),
                    'Electrical Sub Station': bool(data.get('electrical_sub_station'))
                },
                'predicted_psm': f"{predicted_psm:,.2f}",
                'total_price': f"{total_price:,.2f}",
                'room_size': data['room_size'],
                'building_age': data['building_age'],
                'total_units': data['total_units'],
            }

            return render(request, 'predict.html', context)

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return render(request, 'predict.html', {
                'error': f"Prediction error: {str(e)}",
                'districts': District.objects.all().order_by('name'),
                'distance_fields': {
                    'train_station': 0,
                    'airport': 0,
                    'university': 0,
                    'department_store': 0,
                    'hospital': 0
                },
                'facilities': get_facilities()
            })

    return prediction_form(request)

def explore_view(request):
    # Load the dataset from the CSV file
    csv_file_path = 'app_condo/data/condo_data_final.csv'  # Ensure the path is correct
    df = pd.read_csv(csv_file_path)

    # Get unique districts from the 'District' column
    districts = df['District'].unique().tolist()
    
    selected_district = request.GET.get('district', None)

    # Initialize variables for chart data
    labels = []
    data = []

    if selected_district:
        # Filter the DataFrame for the selected district
        district_data = df[df['District'] == selected_district]

        # Create frequency distribution for PSM
        frequency_distribution = district_data['PSM'].value_counts().sort_index()

        # Prepare labels and data for the chart
        labels = frequency_distribution.index.tolist()  # PSM values
        data = frequency_distribution.values.tolist()   # Frequencies

    return render(request, 'explore.html', {
        'districts': districts,
        'selected_district': selected_district,
        'labels': labels,
        'data': data,
    })

def loan_table_view(request):
    schedule = []

    if request.method == "POST":
        try:
            loan_amount = float(request.POST.get("loan_amount", 0))
            years = int(request.POST.get("years", 1))
            interest_rate = float(request.POST.get("interest_rate", 0)) / 100

            if loan_amount <= 0 or years <= 0 or interest_rate < 0:
                return render(request, "loan_table.html", {"error": "Please enter valid loan details."})

            monthly_rate = interest_rate / 12
            payments = years * 12
            payment_amount = loan_amount * (monthly_rate / (1 - (1 + monthly_rate) ** -payments))

            for month in range(1, payments + 1):
                interest_payment = loan_amount * monthly_rate
                principal_payment = payment_amount - interest_payment
                loan_amount -= principal_payment
                schedule.append({
                    "month": month,
                    "principal_payment": round(principal_payment, 2),
                    "interest_payment": round(interest_payment, 2),
                    "total_payment": round(payment_amount, 2),
                    "remaining_balance": round(loan_amount, 2)
                })

            return render(request, "loan_table.html", {"loan_table": schedule})

        except ValueError:
            return render(request, "loan_table.html", {"error": "Invalid input provided."})
        except Exception as e:
            logger.error(f"Error in loan table view: {str(e)}")
            return render(request, "loan_table.html", {"error": f"An error occurred: {str(e)}"})

    return render(request, "loan_table.html")
