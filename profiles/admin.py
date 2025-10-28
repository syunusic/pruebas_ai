from django.contrib import admin

from .models import AlumniProfile, ProfileAuditLog


@admin.register(AlumniProfile)
class AlumniProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'apellido_solo', 'marital_status', 'children_count', 'company_name', 'updated_at')
    list_filter = ('marital_status', 'parents_status')
    search_fields = ('user__full_name', 'user__email', 'apellido_solo', 'company_name')
    ordering = ('apellido_solo', 'user__full_name')
    readonly_fields = ('created_at', 'updated_at', 'consent_timestamp')

    fields = (
        'user',
        'apellido_solo',        # <-- acá
        'parents_status', 'marital_status', 'children_count',
        'job_title', 'company_name', 'company_start_year',
        'job_summary', 'email_visible',
        'consent_given', 'consent_timestamp',
        'created_at', 'updated_at',
    )



@admin.register(ProfileAuditLog)
class ProfileAuditLogAdmin(admin.ModelAdmin):
    list_display = ('profile', 'changed_by', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('profile__user__full_name', 'changed_by__full_name')
    readonly_fields = ('profile', 'changed_by', 'changes', 'created_at')
