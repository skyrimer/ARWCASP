import patrol
import os

def get_boroughs():
    """
    Returns a list of boroughs in London.
    """
    return [
        "Barking and Dagenham",
        "Barnet",
        "Bexley",
        "Brent",
        "Bromley",
        "Camden",
        "Croydon",
        "Ealing",
        "Enfield",
        "Greenwich",
        "Hackney",
        "Hammersmith and Fulham",
        "Haringey",
        "Harrow",
        "Havering",
        "Hillingdon",
        "Hounslow",
        "Islington",
        "Royal Borough of Kensington and Chelsea",
        "Kingston upon Thames",
        "Lambeth",
        "Lewisham",
        "Merton",
        "Newham",
        "Redbridge",
        "Richmond upon Thames",
        "Southwark",
        "Sutton",
        "Tower Hamlets",
        "Waltham Forest",
        "Wandsworth",
        "City of Westminster",
        "City of London"
    ]

def route_exists(location, ward):
    """
    Checks if a patrol route already exists for the given location.
    """
    
    if ward:
        path = os.path.join("ward_routes", f"{location.replace(' ', '_').lower()}.html")
    else:
        path = os.path.join("borough_routes", f"{location.replace(' ', '_').lower()}.html")

    return os.path.exists(path)

def generate_boroughs(override=False):
    """
    Generates the patrol routes for all boroughs.
    """

    boroughs = get_boroughs()
    
    for borough in boroughs:
        if not override and route_exists(borough, False):
            print(f"Patrol route for {borough} already exists. Skipping...")
            continue

        print(f"Generating patrol route for {borough}...")
        patrol.process_location(borough, False)

def get_wards():
    """
    Returns a list of wards in London.
    """
    return [
        # Barking and Dagenham
        "Abbey", "Alibon", "Becontree", "Chadwell Heath", "Eastbrook", "Eastbury", "Gascoigne", "Goresbrook", "Heath", "Longbridge", "Mayesbrook", "Parsloes", "River", "Thames", "Valence", "Village", "Whalebone",
        # Barnet
        "Brunswick Park", "Burnt Oak", "Childs Hill", "Colindale", "Coppetts", "East Barnet", "East Finchley", "Edgware", "Finchley Church End", "Garden Suburb", "Golders Green", "Hale", "Hendon", "High Barnet", "Mill Hill", "Oakleigh", "Totteridge", "Underhill", "West Finchley", "West Hendon", "Woodhouse",
        # Bexley
        "Barnehurst", "Belvedere", "Blackfen and Lamorbey", "Blendon and Penhill", "Bridleway", "Crayford", "Crook Log", "Danson Park", "East Wickham", "Erith", "Falconwood and Welling", "Longlands", "North End", "Sidcup", "Slade Green and Northend", "St Mary's", "St Michael's", "Thamesmead East", "West Heath",
        # Brent
        "Alperton", "Barnhill", "Brondesbury Park", "Dollis Hill", "Dudden Hill", "Fryent", "Harlesden", "Kensal Green", "Kenton", "Kilburn", "Mapesbury", "Northwick Park", "Preston", "Queens Park", "Queensbury", "Stonebridge", "Sudbury", "Tokyngton", "Welsh Harp", "Wembley Central", "Willesden Green",
        # Bromley
        "Bickley", "Biggin Hill", "Bromley Common and Keston", "Bromley Town", "Chelsfield and Pratts Bottom", "Chislehurst", "Clock House", "Copers Cope", "Cray Valley East", "Cray Valley West", "Crystal Palace", "Darwin", "Farnborough and Crofton", "Hayes and Coney Hall", "Kelsey and Eden Park", "Mottingham and Chislehurst North", "Orpington", "Penge and Cator", "Petts Wood and Knoll", "Plaistow and Sundridge", "Shortlands", "West Wickham",
        # Camden
        "Belsize", "Bloomsbury", "Camden Town with Primrose Hill", "Cantelowes", "Fortune Green", "Frognal and Fitzjohns", "Gospel Oak", "Hampstead Town", "Haverstock", "Highgate", "Holborn and Covent Garden", "Kentish Town", "Kilburn", "King's Cross", "Regent's Park", "St Pancras and Somers Town", "Swiss Cottage", "West Hampstead",
        # Croydon
        "Addiscombe", "Ashburton", "Bensham Manor", "Broad Green", "Coulsdon East", "Coulsdon West", "Croham", "Fairfield", "Fieldway", "Heathfield", "Kenley", "New Addington", "Norbury", "Purley", "Sanderstead", "Selhurst", "Selsdon and Ballards", "Shirley", "South Norwood", "Thornton Heath", "Upper Norwood", "Waddon", "West Thornton", "Woodside",
        # Ealing
        "Acton Central", "Cleveland", "Dormers Wells", "Ealing Broadway", "Ealing Common", "East Acton", "Elthorne", "Greenford Broadway", "Greenford Green", "Hanger Hill", "Hobbayne", "Lady Margaret", "North Greenford", "Northfield", "Northolt Mandeville", "Northolt West End", "Perivale", "South Acton", "Southall Broadway", "Southall Green", "Southfield", "Walpole",
        # Enfield
        "Bowes", "Bush Hill Park", "Chase", "Cockfosters", "Edmonton Green", "Enfield Highway", "Enfield Lock", "Grange", "Haselbury", "Highfield", "Jubilee", "Lower Edmonton", "Palmers Green", "Ponders End", "Southbury", "Southgate", "Southgate Green", "Town", "Turkey Street", "Upper Edmonton", "Winchmore Hill",
        # Greenwich
        "Abbey Wood", "Blackheath Westcombe", "Charlton", "Coldharbour and New Eltham", "Eltham North", "Eltham South", "Eltham West", "Glyndon", "Greenwich West", "Kidbrooke with Hornfair", "Middle Park and Sutcliffe", "Peninsula", "Plumstead", "Shooters Hill", "Thamesmead Moorings", "Woolwich Common", "Woolwich Riverside",
        # Hackney
        "Brownswood", "Cazenove", "Clissold", "Dalston", "De Beauvoir", "Hackney Central", "Hackney Downs", "Hackney Wick", "Haggerston", "Hoxton", "King's Park", "Leabridge", "London Fields", "Lordship", "New River", "Queensbridge", "Shacklewell", "Springfield", "Stamford Hill West", "Stoke Newington Central", "Victoria", "Wick",
        # Hammersmith and Fulham
        "Addison", "Askew", "Avonmore and Brook Green", "College Park and Old Oak", "Fulham Broadway", "Fulham Reach", "Hammersmith Broadway", "Munster", "North End", "Palace Riverside", "Parsons Green and Walham", "Ravenscourt Park", "Sands End", "Shepherd's Bush Green", "Town", "Wormholt and White City",
        # Haringey
        "Alexandra", "Bounds Green", "Bruce Grove", "Crouch End", "Fortis Green", "Harringay", "Highgate", "Hornsey", "Muswell Hill", "Noel Park", "Northumberland Park", "Seven Sisters", "St Ann's", "Stroud Green", "Tottenham Green", "Tottenham Hale", "West Green", "White Hart Lane", "Woodside",
        # Harrow
        "Belmont", "Canons", "Centenary", "Greenhill", "Harrow on the Hill", "Harrow Weald", "Hatch End", "Headstone North", "Headstone South", "Kenton East", "Kenton West", "Marlborough", "Pinner", "Pinner South", "Queensbury", "Rayners Lane", "Roxbourne", "Roxeth", "Stanmore Park", "Wealdstone", "West Harrow",
        # Havering
        "Brooklands", "Cranham", "Elm Park", "Emerson Park", "Gooshays", "Hacton", "Harold Wood", "Heaton", "Hylands", "Mawneys", "Pettits", "Rainham and Wennington", "Romford Town", "St Andrew's", "South Hornchurch", "Squirrel's Heath", "Upminster",
        # Hillingdon
        "Barnhill", "Botwell", "Brunel", "Cavendish", "Charville", "Eastcote and East Ruislip", "Harefield", "Heathrow Villages", "Hillingdon East", "Ickenham", "Manor", "Northwood", "Northwood Hills", "Pinkwell", "South Ruislip", "Townfield", "Uxbridge North", "Uxbridge South", "West Drayton", "West Ruislip", "Yeading", "Yiewsley",
        # Hounslow
        "Bedfont", "Brentford", "Chiswick Homefields", "Chiswick Riverside", "Cranford", "Feltham North", "Feltham West", "Hanworth", "Hanworth Park", "Heston Central", "Heston East", "Heston West", "Hounslow Central", "Hounslow East", "Hounslow Heath", "Hounslow South", "Hounslow West", "Isleworth", "Osterley and Spring Grove", "Syon", "Turnham Green",
        # Islington
        "Barnsbury", "Bunhill", "Caledonian", "Canonbury", "Clerkenwell", "Finsbury Park", "Highbury East", "Highbury West", "Hillrise", "Holloway", "Junction", "Mildmay", "St George's", "St Mary's", "St Peter's", "Tollington",
        # Kensington and Chelsea
        "Abingdon", "Brompton and Hans Town", "Campden", "Chelsea Riverside", "Colville", "Courtfield", "Dalgarno", "Earl's Court", "Golborne", "Holland", "Norland", "Notting Dale", "Pembridge", "Queen's Gate", "Redcliffe", "Royal Hospital", "St Helen's", "Stanley",
        # Kingston upon Thames
        "Alexandra", "Berrylands", "Beverley", "Canbury", "Chessington North and Hook", "Chessington South", "Coombe Hill", "Coombe Vale", "Grove", "Norbiton", "Old Malden", "St James", "St Mark's", "Surbiton Hill", "Tolworth and Hook Rise",
        # Lambeth
        "Bishop's", "Brixton Hill", "Clapham Common", "Clapham Town", "Coldharbour", "Ferndale", "Gipsy Hill", "Herne Hill", "Knight's Hill", "Larkhall", "Oval", "Prince's", "St Leonard's", "St Martin's", "Stockwell", "Streatham Hill", "Streatham South", "Streatham Wells", "Thornton", "Tulse Hill", "Vassall",
        # Lewisham
        "Bellingham", "Blackheath", "Brockley", "Catford South", "Crofton Park", "Downham", "Evelyn", "Forest Hill", "Grove Park", "Ladywell", "Lee Green", "Lewisham Central", "New Cross", "Perry Vale", "Rushey Green", "Sydenham", "Telegraph Hill", "Whitefoot",
        # Merton
        "Abbey", "Cannon Hill", "Colliers Wood", "Cricket Green", "Dundonald", "Figge's Marsh", "Graveney", "Hillside", "Lavender Fields", "Longthornton", "Lower Morden", "Merton Park", "Pollards Hill", "Ravensbury", "Raynes Park", "St Helier", "Trinity", "Village", "West Barnes", "Wimbledon Park",
        # Newham
        "Beckton", "Boleyn", "Canning Town North", "Canning Town South", "Custom House", "East Ham Central", "East Ham North", "East Ham South", "Forest Gate North", "Forest Gate South", "Green Street East", "Green Street West", "Little Ilford", "Manor Park", "Plaistow North", "Plaistow South", "Royal Docks", "Stratford and New Town", "Wall End", "West Ham",
        # Redbridge
        "Aldborough", "Barkingside", "Bridge", "Chadwell", "Church End", "Clayhall", "Clementswood", "Cranbrook", "Fairlop", "Fullwell", "Goodmayes", "Hainault", "Loxford", "Mayfield", "Monkhams", "Newbury", "Roding", "Seven Kings", "Snaresbrook", "Valentines", "Wanstead", "Woodford Bridge", "Woodford Green",
        # Richmond upon Thames
        "Barnes", "East Sheen", "Fulwell and Hampton Hill", "Ham, Petersham and Richmond Riverside", "Hampton", "Hampton North", "Hampton Wick", "Heathfield", "Kew", "Mortlake and Barnes Common", "North Richmond", "South Richmond", "South Twickenham", "St Margarets and North Twickenham", "Teddington", "Twickenham Riverside", "West Twickenham", "Whitton",
        # Southwark
        "Borough and Bankside", "Brunswick Park", "Camberwell Green", "Cathedrals", "Chaucer", "College", "East Dulwich", "Faraday", "Grange", "Lane", "Newington", "North Bermondsey", "Nunhead and Queen's Road", "Peckham", "Peckham Rye", "Riverside", "Rotherhithe", "South Bermondsey", "St George's", "Surrey Docks", "Village", "West Dulwich",
        # Sutton
        "Beddington North", "Beddington South", "Belmont", "Carshalton Central", "Carshalton South and Clockhouse", "Cheam", "Nonsuch", "St Helier", "Stonecot", "Sutton Central", "Sutton North", "Sutton South", "Sutton West", "The Wrythe", "Wallington North", "Wallington South", "Wandle Valley", "Worcester Park",
        # Tower Hamlets
        "Bethnal Green", "Blackwall and Cubitt Town", "Bow East", "Bow West", "Bromley North", "Bromley South", "Canary Wharf", "Island Gardens", "Lansbury", "Limehouse", "Mile End", "Poplar", "Shadwell", "Spitalfields and Banglatown", "St Dunstan's", "St Katharine's and Wapping", "St Peter's", "Stepney Green", "Weavers", "Whitechapel",
        # Waltham Forest
        "Cann Hall", "Cathall", "Chingford Green", "Endlebury", "Forest", "Grove Green", "Hale End and Highams Park", "Hatch Lane", "Higham Hill", "High Street", "Hoe Street", "Larkswood", "Lea Bridge", "Leyton", "Leytonstone", "Markhouse", "Valley", "Walthamstow Central", "Walthamstow West", "William Morris", "Wood Street",
        # Wandsworth
        "Balham", "Bedford", "Earlsfield", "East Putney", "Fairfield", "Furzedown", "Graveney", "Latchmere", "Nightingale", "Northcote", "Queenstown", "Roehampton and Putney Heath", "Shaftesbury", "Southfields", "St Mary's Park", "Thamesfield", "Tooting", "Wandsworth Common", "West Hill", "West Putney",
        # Westminster
        "Abbey Road", "Bayswater", "Bryanston and Dorset Square", "Churchill", "Church Street", "Harrow Road", "Hyde Park", "Knightsbridge and Belgravia", "Lancaster Gate", "Little Venice", "Maida Vale", "Marylebone High Street", "Queen's Park", "Regent's Park", "St James's", "Tachbrook", "Vincent Square", "Warwick", "Westbourne", "West End",
        # City of London
        "Aldersgate", "Aldgate", "Bassishaw", "Billingsgate", "Bishopsgate", "Bread Street", "Bridge", "Broad Street", "Candlewick", "Castle Baynard", "Cheap", "Coleman Street", "Cordwainer", "Cornhill", "Cripplegate", "Dowgate", "Farringdon Within", "Farringdon Without", "Langbourn", "Lime Street", "Portsoken", "Queenhithe", "Tower", "Vintry", "Walbrook"
    ]

def generate_wards(override=False):
    """
    Generates the patrol routes for all wards.
    """
    
    wards = get_wards()
    
    for ward in wards:
        if not override and route_exists(ward, True):
            print(f"Patrol route for {ward} already exists. Skipping...")
            continue

        print(f"Generating patrol route for {ward}...")
        patrol.process_location(ward, True)

def check_boroughs():
    """
    Checks if all borough patrol routes exist.
    """
    boroughs = get_boroughs()

    allexist = True
    for borough in boroughs:
        if not route_exists(borough, False):
            print(f"Patrol route for {borough} does not exist.")
            allexist = False
    
    if allexist:
        print("All borough patrol routes exist.")

def create_point_image(location):
    """
    Creates an image of all of the points the patrol route uses.
    """

    patrol.process_location(location, False, True)
    

if __name__ == "__main__":
    #generate_boroughs(True)
    #print("All borough patrol routes generated successfully.")

    #generate_wards()
    #print("All ward patrol routes generated successfully.")

    # Example usage
    #patrol.process_location("City of London")

    #check_boroughs()

    create_point_image("City of London")

    pass


