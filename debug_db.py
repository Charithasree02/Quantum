import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

print(f"URL: {url}")
print(f"Key loaded: {'Yes' if key else 'No'}")

if not url or not key:
    print("Missing credentials")
    exit(1)

supabase = create_client(url, key)

print("\n--- Testing Profiles Table Access ---")
try:
    response = supabase.table('profiles').select("*").execute()
    print(f"Profiles count: {len(response.data)}")
    for p in response.data:
        print(f"User: {p.get('email', 'N/A')} | Code: {p.get('sharing_code')}")
except Exception as e:
    print(f"Error accessing profiles: {e}")

print("\n--- Testing Auth Signup (Test User) ---")
user_id = None
try:
    email = "debug@example.com"
    password = "password"
    
    # Try using admin API to create user (bypasses email confirm usually)
    try:
        # Check if user exists by listing (inefficient but safe for test)
        users = supabase.auth.admin.list_users() 
        existing = next((u for u in users if u.email == email), None) 
        if existing:
            print("User already exists (found via admin).")
            user_id = existing.id
        else:
            print("Creating user via admin.create_user...")
            res = supabase.auth.admin.create_user({
                "email": email,
                "password": password,
                "email_confirm": True
            })
            if res.user:
                print(f"User created via Admin: {res.user.id}")
                user_id = res.user.id
            else:
                 print("Admin create returned no user.")
                 user_id = None
                 
        if user_id:         
            # Manually insert profile if not exists
            try:
                prof = supabase.table('profiles').select("*").eq('id', user_id).execute()
                if not prof.data:
                    print("Inserting profile manually...")
                    supabase.table('profiles').insert({"id": user_id, "email": email, "sharing_code": "DEBUG-ADMIN"}).execute()
                    print("Profile inserted.")
                else:
                    print("Profile already exists.")
            except Exception as e:
                print(f"Error checking/inserting profile: {e}")
                
    except Exception as e:
        print(f"Admin auth error: {e}")
        user_id = None
except Exception as e:
    print(f"Outer error: {e}")

if user_id:
    print("\n--- Testing Transmissions Insert (RLS Check) ---")
    try:
        # Try to insert a dummy transmission
        tx = {
            "sender_id": user_id,
            "receiver_sharing_code": "DEBUG-ADMIN", # Send to self
            "backend_used": "simulator",
            "shots_used": 100,
            "original_image_url": "http://example.com/fake.png",
            "status": "TEST_INSERT"
        }
        res = supabase.table('transmissions').insert(tx).execute()
        if res.data:
            print(f"Transmission inserted successfully: {res.data[0]['id']}")
            # Clean up
            supabase.table('transmissions').delete().eq('id', res.data[0]['id']).execute()
            print("Test transmission deleted.")
        else:
            print("Transmission insert returned empty data (RLS blocked?)")
    except Exception as e:
        print(f"Error inserting transmission: {e}")
