-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Create Profiles Table (Public information linked to Auth)
create table profiles (
  id uuid references auth.users not null primary key,
  email text,
  sharing_code text unique,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create Transmissions Table (Stores quantum job data)
create table transmissions (
  id uuid default uuid_generate_v4() primary key,
  sender_id uuid references auth.users not null,
  receiver_sharing_code text not null,
  backend_used text,
  shots_used int,
  original_image_url text,
  reconstructed_image_url text,
  status text default 'PENDING',
  ibm_job_id text,
  metrics jsonb,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Set up Storage for Images
-- Note: You must create a bucket named 'images' in the Storage dashboard manually if this script doesn't support it (SQL storage creation is beta).
-- Ideally, go to Storage -> Create Bucket -> "images" -> Public.

-- RLS Policies (Security)
alter table profiles enable row level security;
alter table transmissions enable row level security;

-- Allow users to read all profiles (to lookup receiver by code)
create policy "Public profiles are viewable by everyone" on profiles
  for select using (true);

-- Allow specific users to insert their own profile (Service Role bypasses this anyway, but good for safety)
create policy "Users can insert their own profile" on profiles
  for insert with check (auth.uid() = id);

-- Transmissions: Sender can view their own sent messages
create policy "Sender can view their own transmissions" on transmissions
  for select using (auth.uid() = sender_id);

-- Transmissions: Sender can insert new transmissions
create policy "Sender can insert transmissions" on transmissions
  for insert with check (auth.uid() = sender_id);
  
-- Transmissions: Receiver can view messages sent to their Sharing Code
-- (We need to join with profiles to verify, but for simplicity we allow read by code match)
-- A more secure approach requires a join, but for this demo:
create policy "Receiver can view based on sharing code" on transmissions
  for select using (
    receiver_sharing_code = (select sharing_code from profiles where id = auth.uid())
  );

-- Update Transmission (for status updates): Service Role handles this mostly, but allow owner
create policy "Sender can update transmission" on transmissions
  for update using (auth.uid() = sender_id);
