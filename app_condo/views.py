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
import json

csv_file_path = 'app_condo/data/condo_data_explore.csv'

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
        'Swimmingpool', 'CarPark', 'CCTV', 'Fitness',
        'Library', 'Security', 'MiniMart', 'ElectricalSubStation'
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
                'Security': 0 ,
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
                'Fitness', 'Library', 'Swimmingpool' ,'Security' 
                ,'MiniMart', 'ElectricalSubStation'
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
                    'SwimmingPool': bool(data.get('swimming_pool')),
                    'CarPark': bool(data.get('car_park')),
                    'CCTV': bool(data.get('cctv')),
                    'Fitness': bool(data.get('fitness')),
                    'Library': bool(data.get('library')),
                    #'Security': bool(data.get('security')),
                    'MiniMart': bool(data.get('mini_mart')),
                    'ElectricalSubStation': bool(data.get('electrical_sub_station'))
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
    df = pd.read_csv(csv_file_path)
    districts = df['District'].unique().tolist()
    distance_fields = ['TrainStation', 'University', 'Airport', 'Departmentstore', 'Hospital']
    context = {
        'districts': districts,
        'distance_fields': distance_fields,
    }
    return render(request, 'explore.html', context)

def district_psm(request, district):
    df = pd.read_csv(csv_file_path)
    if district:
        # Filter data by the selected district
        district_data = df[df['District'] == district]
        # Group by subdistrict and calculate mean PSM
        subdistrict_psm = district_data.groupby('Subdistrict')['PSM'].mean().reset_index()
        labels = subdistrict_psm['Subdistrict'].tolist()  # Subdistrict names for X-axis
        values = subdistrict_psm['PSM'].tolist()  # Average PSM for Y-axis

        return JsonResponse({'labels': labels, 'values': values})
    return JsonResponse({'error': 'No district specified'})

def distance_psm(request, distance_field):
    df = pd.read_csv(csv_file_path)
    if distance_field:
        distance_data = df.groupby(distance_field)['PSM'].mean().reset_index()
        labels = distance_data[distance_field].tolist()
        values = distance_data['PSM'].tolist()
        return JsonResponse({'labels': labels, 'values': values})
    return JsonResponse({'error': 'No distance field specified'})

def facilities(request, district):
    df = pd.read_csv(csv_file_path)
    if district:
        district_data = df[df['District'] == district]
        facilities = ['CarPark', 'CCTV', 'Fitness', 'Library', 'Swimmingpool', 'MiniMart', 'ElectricalSubStation']
        facility_counts = district_data[facilities].sum().to_dict()
        return JsonResponse({'labels': list(facility_counts.keys()), 'values': list(facility_counts.values())})
    return JsonResponse({'error': 'No district specified'})

def loan_table_view(request):
    predicted_total_price = request.session.get('predicted_total_price', 0)

    # Initialize variables for form data
    loan_amount = None
    interest_rate = None
    years = None
    schedule = None

    if request.method == "POST":
        try:
            # Retrieve values from the form
            loan_amount = float(request.POST.get("loan_amount", 0))
            years = int(request.POST.get("loan_term", 1))
            interest_rate = float(request.POST.get("interest_rate", 0)) / 100

            if loan_amount <= 0 or years <= 0 or interest_rate < 0:
                return render(request, "loan_table.html", {
                    "error": "Please enter valid loan details.",
                    "predicted_total_price": predicted_total_price,
                    "loan_amount": loan_amount,
                    "interest_rate": interest_rate * 100,
                    "loan_term": years
                })

            # Monthly calculations
            monthly_rate = interest_rate / 12
            payments = years * 12
            payment_amount = loan_amount * (monthly_rate / (1 - (1 + monthly_rate) ** -payments))

            # Generate repayment schedule
            schedule = []
            remaining = loan_amount

            for month in range(1, payments + 1):
                interest_payment = remaining * monthly_rate
                principal_payment = payment_amount - interest_payment
                remaining -= principal_payment

                schedule.append({
                    "month": month,
                    "principal_payment": round(principal_payment, 2),
                    "interest_payment": round(interest_payment, 2),
                    "total_payment": round(payment_amount, 2),
                    "remaining_balance": max(0, round(remaining, 2))
                })

            return render(request, "loan_table.html", {
                "loan_table": schedule,
                "predicted_total_price": predicted_total_price,
                "loan_amount": loan_amount,
                "interest_rate": interest_rate * 100,
                "loan_term": years
            })

        except ValueError:
            return render(request, "loan_table.html", {
                "error": "Invalid input provided.",
                "predicted_total_price": predicted_total_price,
                "loan_amount": loan_amount,
                "interest_rate": interest_rate * 100 if interest_rate is not None else '',
                "loan_term": years
            })
        except Exception as e:
            logger.error(f"Error in loan table view: {str(e)}")
            return render(request, "loan_table.html", {
                "error": f"An error occurred: {str(e)}",
                "predicted_total_price": predicted_total_price,
                "loan_amount": loan_amount,
                "interest_rate": interest_rate * 100 if interest_rate is not None else '',
                "loan_term": years
            })

    return render(request, "loan_table.html", {"predicted_total_price": predicted_total_price})